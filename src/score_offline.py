"""Run UNMODIFIED official local_score with local manifest data and our predictor.
Scores every timestep of selected validation shots; no HF/network calls.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from offline_model import predict, inputs_only
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import local_score as official


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=ROOT/'models'/'offline_v3')
    ap.add_argument('--model', default='mlp')
    ap.add_argument('--mode', choices=['model','perfect','zeros'], default='model')
    ap.add_argument('--n-shots', type=int, default=3)
    ap.add_argument('--storage-dtype', choices=['float32','float16'], default='float32')
    args = ap.parse_args()
    manifest = json.loads((args.run/'manifest.json').read_text())
    selected = manifest['validation'][:args.n_shots]
    train_paths = {r['path'] for r in manifest['train']}
    assert all(r['path'] not in train_paths for r in selected)
    assert selected
    model = None
    if args.mode == 'model':
        with (args.run/f'{args.model}.pkl').open('rb') as f: model = pickle.load(f)
    def load_local(n, skip):
        result=[]
        for item in selected:
            row=pq.read_table(item['path']).to_pylist()[0]
            result.append(dict(row=row, psi=np.asarray(row['efit_psirz'],dtype=np.float32),
                               q95=np.asarray(row['efit_q95'],dtype=np.float64),
                               betaN=np.asarray(row['efit_beta_n'],dtype=np.float64)))
        return result
    native_predict=official.predict
    def local_predict(shots, mode, pred_npz):
        if mode!='model': return native_predict(shots,mode,pred_npz)
        result = [predict(inputs_only(s['row']),model) for s in shots]
        for p in result:
            p['psirz'] = p['psirz'].astype(args.storage_dtype).astype(np.float64)
            if not np.isfinite(p['psirz']).all(): raise ValueError('Non-finite flux after storage conversion')
        return result
    native_finalize=official.finalize_machine
    def finalize(acc, ref_stats):
        metrics=native_finalize(acc,ref_stats)
        def clean(v):
            if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)): return [clean(x) for x in v]
            if isinstance(v,np.ndarray): return clean(v.tolist())
            if isinstance(v,np.generic): return clean(v.item())
            if isinstance(v,float) and not np.isfinite(v): return None
            return v
        result={'mode':args.mode,'model':args.model,'storage_dtype':args.storage_dtype,'split_unit':'whole_shot',
                'validation_files':[Path(s['path']).name for s in selected],
                'all_timesteps_scored':True,'network_used':False,'metrics':clean(metrics)}
        suffix='' if args.storage_dtype=='float32' else '_float16'
        (args.run/f'score_{args.model}_{args.mode}_{len(selected)}shots{suffix}.json').write_text(json.dumps(result,indent=2))
        return metrics
    official.load_shots=load_local
    official.predict=local_predict
    official.finalize_machine=finalize
    sys.argv=['local_score.py','--n-shots',str(len(selected)),'--mode',args.mode]
    raise SystemExit(official.main())

if __name__=='__main__': main()
