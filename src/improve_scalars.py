"""Improve ONLY the two value-scalar regressors; preserve flux model and shot split."""
import json,pickle,time,resource
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from offline_model import features

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'models/offline_v3'

def arrays(items):
    xs=[];ss=[]
    for item in items:
        pf=pq.ParquetFile(item['path'])
        # Reading inputs/scalars only avoids materializing magnetic flux target grids.
        from offline_model import INPUT_KEYS
        cols=sorted((INPUT_KEYS | {'efit_q95','efit_beta_n'}) & set(pf.schema_arrow.names))
        row=pf.read(columns=cols).to_pylist()[0]
        xs.append(features(row));ss.append(np.column_stack([row['efit_q95'],row['efit_beta_n']]))
    return np.concatenate(xs),np.concatenate(ss)

start=time.perf_counter()
manifest=json.loads((RUN/'manifest.json').read_text())
with (RUN/'mlp.pkl').open('rb') as f:model=pickle.load(f)
x,s=arrays(manifest['train']);xv,sv=arrays(manifest['validation'])
x=model['x_scaler'].transform(x);xv=model['x_scaler'].transform(xv)
fit_start=time.perf_counter()
reg=MLPRegressor(hidden_layer_sizes=(128,64),max_iter=160,batch_size=512,
                learning_rate_init=.001,random_state=42)
reg.fit(x,(s-model['s_mean'])/model['s_scale'])
fit_seconds=time.perf_counter()-fit_start
old=model['scalar_regressor'].predict(xv)*model['s_scale']+model['s_mean']
new=reg.predict(xv)*model['s_scale']+model['s_mean']
old_r2=[r2_score(sv[:,i],old[:,i]) for i in range(2)]
new_r2=[r2_score(sv[:,i],new[:,i]) for i in range(2)]
model['scalar_regressor']=reg
with (RUN/'mlp_scalar_mlp.pkl').open('wb') as f:pickle.dump(model,f)
report={'validation_shots':len(manifest['validation']),'old_scalar_R2':old_r2,'new_scalar_R2':new_r2,
        'fit_seconds':fit_seconds,'wall_seconds':time.perf_counter()-start,
        'peak_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'flux_model_unchanged':True,'online_score':False}
(RUN/'scalar_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
