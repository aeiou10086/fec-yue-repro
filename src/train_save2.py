"""Fix: standardize PCA coeffs before MLP."""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import load_shots_from_local, build_feature_matrix, TargetPCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

N_SHOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 175
N_PCA = 100

shots = load_shots_from_local(n_shots=N_SHOTS, local_data_dir="../hf_dataset", config="diii_d_train")
X, Y, S, ids = build_feature_matrix(shots, include_thomson=False)
print(f"X={X.shape} Y={Y.shape}", flush=True)

scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
tpca = TargetPCA(n_components=N_PCA).fit(Y)
Z = tpca.transform(Y)
zs = StandardScaler().fit(Z); Zn = zs.transform(Z)
print("PCA evr cum5:", np.cumsum(tpca.explained_variance_ratio)[:5], flush=True)

from sklearn.model_selection import train_test_split
Xtr, Xte, Ztr, Zte = train_test_split(Xs, Zn, test_size=0.1, random_state=0)
mlp = MLPRegressor(hidden_layer_sizes=(512,256,128), activation="relu",
                   max_iter=400, random_state=0, batch_size=1024,
                   learning_rate_init=1e-3, verbose=False)
mlp.fit(Xtr, Ztr)
print("psi-MLP heldout R2:", mlp.score(Xte, Zte), flush=True)

mzs, scal_regs = {}, {}
labels = ["efit_beta_n","efit_li","efit_q95","efit_r_axis","efit_z_axis"]
for name in ["efit_q95", "efit_beta_n"]:
    j = labels.index(name); y = S[:, j]; m = np.isfinite(y)
    mzs[name] = StandardScaler().fit(y[m].reshape(-1,1))
    yn = mzs[name].transform(y[m].reshape(-1,1)).ravel()
    r = MLPRegressor(hidden_layer_sizes=(256,128), max_iter=400, random_state=0)
    r.fit(Xs[m], yn)
    scal_regs[name] = r
    print(name, "train R2:", r.score(Xs[m], yn), flush=True)

out = Path(__file__).resolve().parent.parent / "models"
with open(out/"pca_mlp.pkl","wb") as f:
    pickle.dump({"scaler":scaler,"tpca":tpca,"zscaler":zs,"mlp":mlp,"scal_regs":scal_regs,"mzs":mzs}, f)
print("saved", out/"pca_mlp.pkl")
