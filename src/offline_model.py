"""Offline DIII-D inputs only. Never inspect equilibrium targets during inference."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import D3D_MAGNETICS_SIGNALS, interpolate_magnetics_to_efit
from data_fixes import fix_d3d_ip_times

INPUT_KEYS = {'source', 'efit_times', 'magnetics_time', 'magnetics_plasma_current_times'} | {
    f'magnetics_{s}' for s in D3D_MAGNETICS_SIGNALS
}

def inputs_only(row):
    return {k: row[k] for k in INPUT_KEYS if k in row}

def features(row):
    row = inputs_only(row)
    if row.get('source', 'DIII-D') != 'DIII-D':
        raise ValueError('This baseline supports DIII-D only; MAST needs its own physical mapping.')
    times = np.asarray(row['efit_times'], dtype=np.float64)
    common = np.asarray(row['magnetics_time'], dtype=np.float64)
    fixed_ip = fix_d3d_ip_times(row)
    mag = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        key = f'magnetics_{sig}'
        if key in row:
            mag[sig] = {'times': fixed_ip if sig == 'plasma_current' else common,
                        'values': np.asarray(row[key], dtype=np.float32)}
    x = interpolate_magnetics_to_efit({'efit_times': times, 'magnetics': mag})
    if not np.isfinite(x).all():
        raise ValueError('Non-finite input features; handle explicitly rather than silently dropping test frames.')
    return x

class HybridValueRegressor:
    """Keep the tree q95 head and use the MLP betaN head (same normalization)."""
    def __init__(self, tree, mlp):
        self.tree, self.mlp = tree, mlp

    def predict(self, x):
        return np.column_stack([self.tree.predict(x)[:, 0], self.mlp.predict(x)[:, 1]])


def predict_features(x, model):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(D3D_MAGNETICS_SIGNALS) or not np.isfinite(x).all():
        raise ValueError('Expected finite (T, 21) input feature matrix')
    x = model['x_scaler'].transform(x)
    if model.get('ensemble') and 'psi_regressors' in model:
        z = np.mean([m.predict(x) for m in model['psi_regressors']], axis=0) * model['z_scale'] + model['z_mean']
    else:
        z = model['psi_regressor'].predict(x) * model['z_scale'] + model['z_mean']
    psi = model['pca'].inverse_transform(z).reshape(-1, 65, 65)
    scal = model['scalar_regressor'].predict(x) * model['s_scale'] + model['s_mean']
    return {'psirz': psi.astype(np.float32), 'q95': scal[:, 0].astype(np.float32),
            'betaN': scal[:, 1].astype(np.float32)}


def predict(row, model):
    return predict_features(features(row), model)
