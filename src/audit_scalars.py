"""Audit scalar heads on new whole shots, disjoint from both training and tuning sets.
No fitting or hyperparameter search here. Keeps audit IDs/hash evidence.
"""
import json,pickle,time,hashlib,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import r2_score
from offline_model import INPUT_KEYS,features
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'models/offline_v3'
manifest=json.loads((RUN/'manifest.json').read_text())
exclude={x['path'] for k in ['train','validation'] for x in manifest[k]}
paths=[p for p in sorted(Path(manifest['source_dir']).glob('*.parquet')) if str(p) not in exclude]
chosen=np.random.RandomState(702).choice(paths,size=min(24,len(paths)),replace=False)
xs=[];ys=[];records=[];start=time.perf_counter()
for path in chosen:
    path=Path(path)
    pf=pq.ParquetFile(path)
    row=pf.read(columns=sorted((INPUT_KEYS|{'efit_q95','efit_beta_n'})&set(pf.schema_arrow.names))).to_pylist()[0]
    xs.append(features(row));ys.append(np.column_stack([row['efit_q95'],row['efit_beta_n']]))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    records.append({'path':str(path),'sha256':h.hexdigest()})
    print('audit loaded',len(records),flush=True)
x=np.concatenate(xs);y=np.concatenate(ys)
report={'audit_shots':len(records),'audit_frames':len(x),'disjoint_from_training_and_tuning':True,
        'audit_records':records,'metrics':{},'online_score':False}
for name in ['mlp','mlp_hybrid']:
    with (RUN/f'{name}.pkl').open('rb') as f:model=pickle.load(f)
    p=model['scalar_regressor'].predict(model['x_scaler'].transform(x))*model['s_scale']+model['s_mean']
    r=[float(r2_score(y[:,i],p[:,i])) for i in range(2)]
    report['metrics'][name]={'R2_q95':r[0],'R2_betaN':r[1],'mean_scalar_R2':float(np.mean(r))}
report['wall_seconds']=time.perf_counter()-start
(RUN/'scalar_new_shot_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='audit_records'}),flush=True)
