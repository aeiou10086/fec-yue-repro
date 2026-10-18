"""Build a streaming NPZ from cached, indexed DIII-D inputs; fail closed on gaps.
Default is a FULL public-test artifact; --smoke-shots is explicitly not submittable.
No network, credentials, model uploads, or public code publishing.
"""
import argparse,hashlib,json,pickle,sys,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STARTER=ROOT/'fusion-equilibrium-challenge-starter'
sys.path.insert(0,str(STARTER/'my_experiments'))
from offline_model import predict_features


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,default=STARTER/'models/offline_v3/mlp_hybrid.pkl')
    ap.add_argument('--dtype',choices=['float16','float32'],default='float32')
    ap.add_argument('--smoke-shots',type=int,default=0)
    args=ap.parse_args()
    src=ROOT/'tmp/d3d_public_manifest.json'
    manifest=json.loads(src.read_text()); entries=sorted(manifest['files'],key=lambda x:x['path'])
    order=json.loads((ROOT/'tmp/order_validation.json').read_text())
    assert order['manifest_sha256']==sha(src) and len(entries)==874
    assert order['revision']==manifest['revision']
    cache=ROOT/'hf_dataset/input_cache/diii_d_public_test'
    chosen=list(enumerate(entries))
    if args.smoke_shots:
        chosen=[x for x in chosen if (cache/f'{x[0]:04d}.json').exists()][:args.smoke_shots]
    assert chosen
    records=[]
    for i,item in chosen:
        f=cache/f'{i:04d}.npz'; m=cache/f'{i:04d}.json'
        if not (f.exists() and m.exists()):
            raise SystemExit(f'Not ready: missing public shot index {i}. No submission produced.')
        metadata=json.loads(m.read_text())
        assert metadata['index']==i and metadata['source_path']==item['path']
        assert metadata['revision']==manifest['revision']
        assert metadata['cache_sha256']==sha(f)
        records.append(metadata)
    with args.model.open('rb') as f:model=pickle.load(f)
    dirname='smoke_ONLY' if args.smoke_shots else 'development_candidate'
    outdir=ROOT/'artifacts'/dirname;outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/'diii_d_public_test.npz'
    temp=out.with_suffix('.npz.tmp'); shapes={};start=time.perf_counter()
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for (i,_),meta in zip(chosen,records):
            with np.load(cache/f'{i:04d}.npz',allow_pickle=False) as inp:
                x=inp['X'];times=inp['times']
            assert len(times)==meta['frames'] and x.shape==(len(times),21)
            pred=predict_features(x,model)
            for k,v in pred.items():
                a=v.astype(args.dtype if k=='psirz' else 'float32')
                assert np.isfinite(a).all()
                assert a.shape==((len(times),65,65) if k=='psirz' else (len(times),))
                key=f'shot_{i:04d}_{k}';shapes[key]=list(a.shape)
                with z.open(key+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,a,allow_pickle=False)
            if len(shapes)%150==0:print(f'Built {len(shapes)//3}/{len(chosen)} shots',flush=True)
    temp.replace(out)
    # Reopen every array: serialization/CRC/shape/finiteness validation.
    with np.load(out,allow_pickle=False) as z:
        assert set(z.files)==set(shapes)
        for key,shape in shapes.items():
            a=z[key];assert list(a.shape)==shape and np.isfinite(a).all()
    report={'artifact':str(out),'sha256':sha(out),'model_sha256':sha(args.model),
            'revision':manifest['revision'],'shots':len(chosen),'frames':sum(r['frames'] for r in records),
            'full_public_test':len(chosen)==874 and not args.smoke_shots,'dtype_psi':args.dtype,
            'serialization_validation_pass':True,'network_used':False,'submitted':False,
            'wall_seconds':time.perf_counter()-start,'bytes':out.stat().st_size,
            'scope':'Ch1 DIII-D only; no MAST prediction or Ch2 claim'}
    (outdir/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
