"""Train reproducible, shot-disjoint DIII-D baselines from existing parquet files.
No downloads, uploads or test labels. Quarantined files stay untouched.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from offline_model import features, predict, inputs_only
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def load(paths):
    xs, ys, ss = [], [], []
    for i, path in enumerate(paths):
        row = pq.read_table(path).to_pylist()[0]
        x = features(row)
        y = np.asarray(row['efit_psirz'], dtype=np.float32).reshape(len(x), -1)
        s = np.column_stack([row['efit_q95'], row['efit_beta_n']]).astype(np.float32)
        if not np.isfinite(y).all() or not np.isfinite(s).all():
            raise ValueError(f'Nonfinite training labels: {path.name}')
        xs.append(x); ys.append(y); ss.append(s)
        if i % 25 == 0: print(f'loaded {i + 1}/{len(paths)}', flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-shots', type=int, default=160)
    ap.add_argument('--valid-shots', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=Path, default=ROOT/'models'/'offline_v3')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = ROOT.parent/'hf_dataset'/'data'/'diii_d_train'
    good, bad = [], []
    for p in sorted(source.glob('*.parquet')):
        try:
            meta = pq.ParquetFile(p)
            if meta.metadata.num_rows != 1: raise ValueError('expected one shot per file')
            needed = {'efit_psirz', 'efit_times', 'efit_q95', 'efit_beta_n'}
            if not needed.issubset(meta.schema_arrow.names): raise ValueError('missing targets')
            good.append(p)
        except Exception as e:
            bad.append({'file': p.name, 'reason': type(e).__name__})
    if len(good) < args.train_shots + args.valid_shots: raise ValueError('Insufficient validated local files')
    order = np.random.RandomState(args.seed).permutation(len(good))
    va = [good[i] for i in order[:args.valid_shots]]
    tr = [good[i] for i in order[args.valid_shots:args.valid_shots + args.train_shots]]
    assert not set(va).intersection(tr)
    manifest = {'seed': args.seed, 'source_dir': str(source), 'split_unit': 'whole_shot',
                'network_used': False, 'invalid_files': bad,
                'train': [{'path': str(p), 'sha256': digest(p)} for p in tr],
                'validation': [{'path': str(p), 'sha256': digest(p)} for p in va]}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Validated {len(good)} files; excluded {len(bad)}. Train {len(tr)}, validation {len(va)}.', flush=True)
    x, y, s = load(tr)
    xv, yv, sv = load(va)
    xs = StandardScaler().fit(x)
    x = xs.transform(x); xv_scaled = xs.transform(xv)
    pca = PCA(n_components=32, svd_solver='randomized', random_state=args.seed).fit(y)
    z = pca.transform(y)
    zm = z.mean(axis=0)
    # Do not give tiny/noisy PCA components the same weight as the leading components.
    zs = np.maximum(z.std(axis=0), z.std(axis=0).max() * .01)
    sm, ss = s.mean(axis=0), np.maximum(s.std(axis=0), 1e-6)
    base = dict(x_scaler=xs, pca=pca, z_mean=zm, z_scale=zs, s_mean=sm, s_scale=ss,
                manifest='manifest.json', source='DIII-D', input_schema='coil currents + corrected Ip')
    print(f'frames train={len(x)}, validation={len(xv)}; PCA variance={pca.explained_variance_ratio_.sum():.6f}', flush=True)
    scalar = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=3, random_state=args.seed)
    scalar.fit(x, (s-sm)/ss)
    reports = {}
    for name, reg in [
        ('extra_trees', ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=3, random_state=args.seed)),
        ('mlp', MLPRegressor(hidden_layer_sizes=(256,128), max_iter=120, batch_size=512,
                            learning_rate_init=.001, random_state=args.seed)),
    ]:
        print(f'Training {name}', flush=True)
        reg.fit(x, (z-zm)/zs)
        model = dict(base, psi_regressor=reg, scalar_regressor=scalar)
        yh = pca.inverse_transform(reg.predict(xv_scaled)*zs+zm)
        sh = scalar.predict(xv_scaled)*ss+sm
        reports[name] = {'validation_unit': 'whole_shot', 'n_shots': len(va), 'n_frames': len(xv),
                         'r2_psi_global': float(r2_score(yv.ravel(), yh.ravel())),
                         'mae_psi': float(mean_absolute_error(yv.ravel(), yh.ravel())),
                         'r2_q95': float(r2_score(sv[:,0], sh[:,0])),
                         'r2_betaN': float(r2_score(sv[:,1], sh[:,1]))}
        with (args.out/f'{name}.pkl').open('wb') as f: pickle.dump(model, f)
        print(name, json.dumps(reports[name]), flush=True)
        # Regression test: inference must work with no labels and ignore poisoned target fields.
        raw = pq.read_table(va[0]).to_pylist()[0]
        a = predict(inputs_only(raw), model)
        poisoned = dict(raw, efit_psirz='DO NOT READ', efit_q95='DO NOT READ',
                        efit_beta_n='DO NOT READ', magnetics_dsep='DO NOT READ')
        b = predict(poisoned, model)
        for k in a:
            assert np.isfinite(a[k]).all()
            np.testing.assert_array_equal(a[k], b[k])
        assert a['psirz'].shape == (len(raw['efit_times']), 65, 65)
        reports[name]['target_isolation_test'] = 'passed'
        del yh
        (args.out/'validation_metrics.json').write_text(json.dumps(reports, indent=2))
    # Smoke-test on actual inputs-only public data (not on held-out labels).
    pubs = sorted((source.parent/'diii_d_public_test').glob('*.parquet'))
    for p in pubs:
        try: row = pq.read_table(p).to_pylist()[0]
        except Exception: continue
        if 'efit_psirz' in row: raise AssertionError('unexpected public test labels')
        out = predict(row, model)
        np.savez_compressed(args.out/'public_smoke.npz', **{f'shot_0000_{k}': v for k,v in out.items()})
        (args.out/'public_smoke.json').write_text(json.dumps({'file': p.name, 'n_frames': len(row['efit_times']),
                  'finite': all(bool(np.isfinite(v).all()) for v in out.values()),
                  'scope': 'One local public file only; NOT a complete or indexed competition submission'}))
        print('Public input-only smoke test passed. NOT uploaded.', flush=True)
        break
    (args.out/'DONE').write_text('offline training and smoke checks finished\n')

if __name__ == '__main__': main()
