"""
Baselines for the COVERT comparison (matches manuscript/ Table 1 rows):
  * centralized  - COVERT non-private (dp=0): the lossless upper bound (== pooled, see go/no-go).
  * fbttr        - federated block-term WITHOUT DP and WITHOUT sparsity (the prior federated
                   strategy; COVERT minus its two deltas). A clear ablation/baseline.
  * p3ls         - privacy-preserving PLS: shared sample-mode score, per-party LINEAR (rank-1)
                   loadings on flattened features (the masked-linear-algebra line).
  * concat       - naive: PLS on all parties' features concatenated (ignores vertical structure).
  * single       - best single party (P3LS on one party alone).
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from core.covert import fit_covert, predict_covert, _ridge_solve, _as3


def _flatten(parties):
    return [_as3(X).reshape(_as3(X).shape[0], -1) for X in parties]


def fit_p3ls(parties, y, K=2, ridge=1e-2):
    """Vertical PLS by deflation: shared score = sum_p F_p w_p (linear, rank-1 per comp)."""
    F = _flatten(parties)
    P = len(F)
    ymean = float(np.mean(y))
    r = np.asarray(y, float) - ymean
    comps = [[] for _ in range(P)]
    for _ in range(K):
        scores = []
        for p in range(P):
            w = F[p].T @ r
            w /= (norm(w) + 1e-12)
            comps[p].append(w)
            scores.append(F[p] @ w)
        t = np.sum(scores, axis=0)
        g = (t @ r) / (t @ t + 1e-9)
        r = r - g * t
    Z = _p3ls_scores(F, comps)
    beta = _ridge_solve(Z.T @ Z, Z.T @ (np.asarray(y, float) - ymean), ridge)
    return {'comps': comps, 'beta': beta, 'ymean': ymean, 'K': K}


def _p3ls_scores(F, comps):
    K = len(comps[0])
    cols = [np.sum([F[p] @ comps[p][k] for p in range(len(F))], axis=0) for k in range(K)]
    return np.column_stack(cols)


def predict_p3ls(model, parties):
    Z = _p3ls_scores(_flatten(parties), model['comps'])
    return Z @ model['beta'] + model['ymean']


# --- thin wrappers over the COVERT core for the federated-tensor baselines --------------
def fit_centralized(parties, y, R=2, L=2, ridge='auto', seed=0):
    """Non-private, dense (sparsity off): the lossless upper bound."""
    return fit_covert(parties, y, R=R, L=L, ridge=ridge, dp_sigma=0.0,
                      clip=None, sparse_lambda=0.0, seed=seed)


def fit_fbttr(parties, y, R=2, L=2, ridge='auto', seed=0):
    """Prior federated block-term: no DP, no sparse core (COVERT minus its two deltas)."""
    return fit_covert(parties, y, R=R, L=L, ridge=ridge, dp_sigma=0.0,
                      clip=None, sparse_lambda=0.0, seed=seed)


def fit_single_best(parties, y, K=2, metric_fn=None):
    """Best single party under P3LS (fit each party alone; caller scores on val)."""
    return [fit_p3ls([Xp], y, K=K) for Xp in parties]


def fit_concat(parties, y, K=2):
    cat = np.concatenate([_as3(X).reshape(_as3(X).shape[0], -1) for X in parties], axis=1)
    return fit_p3ls([cat], y, K=K)


def predict_concat(model, parties):
    cat = np.concatenate([_as3(X).reshape(_as3(X).shape[0], -1) for X in parties], axis=1)
    return predict_p3ls(model, [cat])
