"""
DWI frozen-embedding SANITY probe (NOT a manuscript number).

Question: do the frozen DINO-backbone DWI embeddings (hcc_embeddings_dwi.npz, (N,4,384)) carry
incident-HCC signal near HCCNet's ~0.71 DWI frozen-baseline regime? Grouped-by-patient CV.

  * Linear probe  = standardize + L2 logistic on the flattened (N, 4*384) embedding -> the standard
    frozen-backbone probe, directly comparable to a ~0.71 frozen baseline.
  * COVERT single-party = the rank-(L,L,1) block-term score on the (N,4,384) tensor (optional; confirms
    the COVERT machinery runs on real embeddings and is at PARITY with the linear probe -- the paper's claim).

Pooled out-of-fold AUROC + PR-AUC (more stable than per-fold means with ~42 positives).
Run from the package root: `python -m experiments.sanity_dwi_probe [npz_path]`.
"""
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    'COVERT_HCC_FEATURES', os.path.expanduser('~/projects/covert/hcc_embeddings_dwi.npz'))
N_SPLITS, SEED = 5, 0

z = np.load(PATH, allow_pickle=True)
dwi, y = z['dwi'].astype(float), z['y'].astype(int)
groups = z['groups']
N = len(y)
print(f"[sanity] {PATH}\n  N={N}  pos={int(y.sum())} ({y.mean():.3f})  "
      f"patients={len(set(groups.tolist()))}  dwi={dwi.shape}", flush=True)

# --- 1) linear probe (frozen-baseline probe) -------------------------------------------------
X = dwi.reshape(N, -1)
oof = np.zeros(N)
skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_auc = []
for tr, va in skf.split(X, y, groups):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
    clf.fit(sc.transform(X[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X[va]))[:, 1]
    oof[va] = p
    if len(set(y[va].tolist())) == 2:
        fold_auc.append(roc_auc_score(y[va], p))
print(f"  [linear probe] pooled OOF AUROC={roc_auc_score(y, oof):.3f}  "
      f"PR-AUC={average_precision_score(y, oof):.3f}  "
      f"(per-fold AUROC mean={np.mean(fold_auc):.3f} +/- {np.std(fold_auc):.3f}, n={len(fold_auc)})",
      flush=True)

# --- 2) COVERT single-party block-term score (optional; parity check) ------------------------
try:
    from core.covert import fit_covert, predict_covert
    oofc = np.zeros(N)
    for tr, va in skf.split(X, y, groups):
        # standardize per (volume,feature) on train stats; apply to both
        mu = dwi[tr].mean(0, keepdims=True); sd = dwi[tr].std(0, keepdims=True)
        sdz = np.where(sd < 1e-8, 1.0, sd)
        Xtr, Xva = (dwi[tr] - mu) / sdz, (dwi[va] - mu) / sdz
        model = fit_covert([Xtr], y[tr].astype(float), R=2, L=2, ridge=1.0, seed=SEED)
        oofc[va] = predict_covert(model, [Xva])
    print(f"  [COVERT 1-party R=2,L=2] pooled OOF AUROC={roc_auc_score(y, oofc):.3f}  "
          f"PR-AUC={average_precision_score(y, oofc):.3f}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"  [COVERT 1-party] skipped ({type(e).__name__}: {e})", flush=True)

print("[sanity] done -- validation only, not a manuscript number.", flush=True)
