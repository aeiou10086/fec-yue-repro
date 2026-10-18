"""Seed-ensemble the zfloor psi MLP (3 seeds) + hybrid scalar head; score 12-shot official."""
import json, pickle, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import sys
STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER / "my_experiments"))
from offline_model import INPUT_KEYS, features, predict_features

RUN = STARTER / "models" / "offline_v3"
manifest = json.loads((RUN / "manifest.json").read_text())
val_paths = [x["path"] for x in manifest["validation"]]

def arrays(paths):
    xs, ys = [], []
    for p in paths:
        pf = pq.ParquetFile(p)
        cols = sorted((INPUT_KEYS | {"efit_psirz"}) & set(pf.schema_arrow.names))
        row = pf.read(columns=cols, use_threads=False).to_pylist()[0]
        x = features(row)
        xs.append(x)
        ys.append(np.asarray(row["efit_psirz"], dtype=np.float32).reshape(len(x), -1))
    return np.concatenate(xs), np.concatenate(ys)

all_paths = sorted(Path(manifest["source_dir"]).glob("*.parquet"))
excl = set(val_paths)
train_all = [str(p) for p in all_paths if str(p) not in excl]
print(f"loading {len(train_all)} shots...", flush=True)
X, Y = arrays(train_all)
xs_ = StandardScaler().fit(X); Xn = xs_.transform(X)
pca = PCA(n_components=32, svd_solver="randomized", random_state=42).fit(Y)
Z = pca.transform(Y); zm = Z.mean(axis=0); zsc = np.maximum(Z.std(axis=0), Z.std(axis=0).max() * .01)
mlps = []
for seed in (42, 7, 2026, 13, 99):
    t0 = time.perf_counter()
    m = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=120, batch_size=512,
                     learning_rate_init=1e-3, random_state=seed)
    m.fit(Xn, (Z - zm) / zsc)
    mlps.append(m)
    print(f"seed {seed} trained {time.perf_counter()-t0:.0f}s", flush=True)
# scalar head from the frozen hybrid (proven best)
h = pickle.load(open(RUN / "mlp_hybrid.pkl", "rb"))
model = {"x_scaler": xs_, "pca": pca, "psi_regressors": mlps, "z_mean": zm, "z_scale": zsc,
         "s_mean": h["s_mean"], "s_scale": h["s_scale"], "scalar_regressor": h["scalar_regressor"],
         "source": "DIII-D", "input_schema": "zfloor 3-seed MLP ensemble + hybrid scalars",
         "ensemble": True}
out = RUN / "zfloor_ens5.pkl"
with out.open("wb") as f: pickle.dump(model, f)
print("saved", out)
