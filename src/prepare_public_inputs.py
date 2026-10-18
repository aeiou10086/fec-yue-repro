"""Prepare DIII-D test inputs only, with a persistent conservative proxy-byte cap.
Uses existing validated full parquets first, otherwise HTTP Range column projection.
No HF token is read or sent. No labels fetched. Never uploads anything.
"""
import argparse
import io
import json
import hashlib
import os
from pathlib import Path
import sys
import threading
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import ssl
from requests.adapters import HTTPAdapter
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
STARTER=ROOT/'fusion-equilibrium-challenge-starter'
sys.path.insert(0,str(STARTER/'my_experiments'))
from offline_model import INPUT_KEYS,features

class TLS12Adapter(HTTPAdapter):
    """Compatibility mode for the proxy path; certificate verification stays enabled."""
    @staticmethod
    def context():
        ctx=ssl.create_default_context(cafile=requests.certs.where())
        ctx.minimum_version=ssl.TLSVersion.TLSv1_2
        ctx.maximum_version=ssl.TLSVersion.TLSv1_2
        return ctx
    def init_poolmanager(self,*args,**kwargs):
        kwargs['ssl_context']=self.context()
        return super().init_poolmanager(*args,**kwargs)
    def proxy_manager_for(self,proxy,**kwargs):
        kwargs['ssl_context']=self.context()
        return super().proxy_manager_for(proxy,**kwargs)

class BudgetExhausted(RuntimeError): pass
class DownloadPaused(RuntimeError): pass

class Budget:
    def __init__(self,path,limit,deadline=None):
        self.path=path; self.lock=threading.Lock(); self.limit=limit; self.paused=False; self.deadline=deadline
        self.state=json.loads(path.read_text()) if path.exists() else {
            'charged_bytes':0,'received_bytes':0,'requests':0,
            'policy':'Reserve requested response bytes BEFORE each range request, including retries. Conservative upper bound.'}
    def reserve(self,n):
        with self.lock:
            if self.paused: raise DownloadPaused('Remote server requested a pause')
            if self.deadline and time.time()>=self.deadline: raise DownloadPaused('Operator execution deadline reached')
            if self.state['charged_bytes']+n>self.limit: raise BudgetExhausted('5GB authorization cap reached')
            self.state['charged_bytes']+=n; self.state['requests']+=1
            self.save()
    def record(self,n):
        with self.lock:
            self.state['received_bytes']+=n
            self.save()
    def save(self):
        self.state['limit_bytes']=self.limit
        t=self.path.with_suffix('.tmp'); t.write_text(json.dumps(self.state,indent=2)); t.replace(self.path)

class RangeFile(io.RawIOBase):
    def __init__(self,url,size,budget):
        self.url=url; self.size=size; self.pos=0; self.budget=budget; self.cache=[]
        self.reset_session()
    def reset_session(self):
        if hasattr(self,'session'): self.session.close()
        self.session=requests.Session(); self.session.trust_env=False
        self.session.proxies={'https':'http://127.0.0.1:7897'}
        self.session.mount('https://',TLS12Adapter())
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0: raise ValueError('negative offset')
        return self.pos
    def get(self,start,end):
        for a,b,data in self.cache:
            if a<=start and end<=b: return data[start-a:end-a]
        n=end-start
        if n<=0: return b''
        for attempt in range(2):
            self.budget.reserve(n)
            try:
                with self.session.get(self.url,headers={'Range':f'bytes={start}-{end-1}',
                     'Accept-Encoding':'identity'},stream=True,timeout=(15,60)) as r:
                    if r.status_code in (403,429):
                        self.budget.paused=True
                        raise DownloadPaused(f'HTTP {r.status_code}; Retry-After={r.headers.get("Retry-After", "not specified")}')
                    if r.status_code!=206: raise RuntimeError(f'Expected HTTP 206, got {r.status_code}')
                    if not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end-1}/'):
                        raise RuntimeError('Unexpected Content-Range')
                    # Reuse this file's verified signed CDN URL; avoid repeated origin redirects.
                    self.url=r.url
                    chunks=[]; got=0
                    for chunk in r.iter_content(65536):
                        got+=len(chunk); self.budget.record(len(chunk))
                        if got>n: raise RuntimeError('Server exceeded requested range')
                        chunks.append(chunk)
                    if got!=n: raise RuntimeError('Truncated range')
                data=b''.join(chunks); self.cache.append((start,end,data)); return data
            except (BudgetExhausted,DownloadPaused): raise
            except Exception:
                self.reset_session()  # discard a failed TLS/keep-alive connection before one retry
                if attempt==1: raise
        raise AssertionError('unreachable')
    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.pos+size)
        data=self.get(self.pos,end); self.pos=end; return data
    def close(self):
        self.session.close(); super().close()


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit-bytes',type=int,default=4_999_900_000)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--max-files',type=int,default=0)
    ap.add_argument('--only-index',type=int)
    ap.add_argument('--deadline-epoch',type=float)
    args=ap.parse_args()
    info=json.loads((ROOT/'tmp/d3d_public_manifest.json').read_text())
    if 'rel="next"' in info.get('pagination_link',''): raise ValueError('Manifest pagination is incomplete')
    entries=sorted(info['files'],key=lambda x:x['path'])
    assert len(entries)==874 and len({x['path'] for x in entries})==874
    cache=ROOT/'hf_dataset'/'input_cache'/'diii_d_public_test'; cache.mkdir(parents=True,exist_ok=True)
    budget=Budget(ROOT/'tmp/proxy_budget_5gb.json',args.limit_bytes,args.deadline_epoch)
    errors=[]; completed=[]
    def process(pair):
        idx,item=pair
        out=cache/f'{idx:04d}.npz'
        meta=cache/f'{idx:04d}.json'
        if meta.exists() and out.exists():
            saved=json.loads(meta.read_text())
            if saved.get('source_path')==item['path'] and saved.get('revision')==info['revision']:
                with np.load(out,allow_pickle=False) as z:
                    assert z['X'].shape==(len(z['times']),21) and np.isfinite(z['X']).all()
                return saved
        local=ROOT/'hf_dataset'/item['path']
        local_ok=local.exists() and local.stat().st_size==item['size']
        if local_ok and item.get('lfs',{}).get('oid'):
            local_ok=sha256(local)==item['lfs']['oid']
        mode='local_sha256_verified' if local_ok else 'http_range_projection'
        if local_ok:
            schema=pq.ParquetFile(local).schema_arrow.names
            row=pq.read_table(local,columns=sorted(INPUT_KEYS & set(schema))).to_pylist()[0]
        else:
            url=f"https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge/resolve/{info['revision']}/{item['path']}"
            with RangeFile(url,item['size'],budget) as rf:
                pf=pq.ParquetFile(rf)
                if pf.metadata.num_rows!=1: raise ValueError('expected exactly one shot')
                if 'efit_psirz' in pf.schema_arrow.names: raise ValueError('public data unexpectedly contains targets')
                intervals=[]
                for gi in range(pf.metadata.num_row_groups):
                    rg=pf.metadata.row_group(gi)
                    for ci in range(rg.num_columns):
                        col=rg.column(ci)
                        if col.path_in_schema.split('.')[0] in INPUT_KEYS:
                            a=min(v for v in (col.dictionary_page_offset,col.data_page_offset) if v is not None and v>=0)
                            intervals.append((a,a+col.total_compressed_size))
                merged=[]
                for a,b in sorted(intervals):
                    if merged and a<=merged[-1][1]+16384:
                        merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
                    else: merged.append((a,b))
                for a,b in merged: rf.get(a,b)
                row=pf.read(columns=sorted(INPUT_KEYS & set(pf.schema_arrow.names)),use_threads=False).to_pylist()[0]
        x=features(row); times=np.asarray(row['efit_times'],dtype=np.float64)
        assert x.shape==(len(times),21)
        temp=out.with_suffix('.tmp')
        with temp.open('wb') as f: np.savez_compressed(f,X=x,times=times)
        temp.replace(out)
        record={'index':idx,'source_path':item['path'],'source_sha256':item.get('lfs',{}).get('oid'),
                'revision':info['revision'],'frames':len(times),'mode':mode,'cache_sha256':sha256(out)}
        meta.write_text(json.dumps(record,indent=2))
        return record
    chosen=list(enumerate(entries))
    if args.max_files: chosen=chosen[:args.max_files]
    if args.only_index is not None: chosen=[chosen[args.only_index]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(process,p):p for p in chosen}
        for task in as_completed(pending):
            idx,item=pending[task]
            try: completed.append(task.result())
            except (BudgetExhausted,DownloadPaused) as e:
                errors.append({'index':idx,'file':item['path'],'error':type(e).__name__,'message':str(e)})
                for other in pending: other.cancel()
            except Exception as e:
                message=re.sub(r'https?://\S+', '<URL>', str(e))[:160]
                errors.append({'index':idx,'file':item['path'],'error':type(e).__name__,'message':message})
            if errors:
                (ROOT/'tmp/public_input_errors.json').write_text(json.dumps(errors,indent=2))
            if (len(completed)+len(errors))%25==0 or (errors and len(errors)==1):
                print(f'cached={len(completed)} errors={len(errors)} charged={budget.state["charged_bytes"]}',flush=True)
    result={'complete':len(completed)==874 and not errors,'expected':874,'cached':len(completed),
            'errors':errors,'budget':budget.state,'order':'pinned manifest lexicographic (verify with HF builder before submission)'}
    (ROOT/'tmp/public_input_progress.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='errors'}), 'error_count=',len(errors),flush=True)
    return 0 if result['complete'] or ((args.max_files or args.only_index is not None) and not errors) else 1

if __name__=='__main__': raise SystemExit(main())
