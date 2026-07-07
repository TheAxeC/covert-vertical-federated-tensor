"""
COVERT core - vertical federated sparse block-term tensor regression with DP on the
aggregated coupling score. Validated on synthetic data (the sibling covert-theory go/no-go):
  - federation lossless at sigma=0 (== centralized),
  - genuine multilinear advantage in the low-rank regime,
  - graceful O(sigma^2) DP degradation.

Per-party feature tensors are 3-way (n_patients x mode_a x mode_b) for genuine multiway views
(ECG: patient x lead x time-feature; MRI: patient x sequence x spatial-feature) or 2-way
(n_patients x n_features) for tabular parties - passed as (n, d, 1), in which case the block
weight is a vector (a linear party, which is the correct behaviour for tabular data).

Only the aggregated, clipped, Gaussian-noised m-vector score crosses the party boundary
(plus the label-residual direction); see Algorithm 1 (Route B) in the manuscript.
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import svd, lstsq, norm


def _as3(X):
    """Coerce a party array to (n, d1, d2): tabular (n, d) -> (n, d, 1)."""
    X = np.asarray(X, dtype=float)
    return X[:, :, None] if X.ndim == 2 else X


def _ridge_solve(G, b, ridge_rel):
    lam = ridge_rel * (np.trace(G) / G.shape[0] + 1e-12)
    sol, *_ = lstsq(G + lam * np.eye(G.shape[0]), b, rcond=None)
    return sol


_RIDGE_GRID = (1.0, 10.0, 100.0, 1e3, 1e4)


def _als_weight(X, r, L, ridge, iters):
    """ALS for the rank-L weight W (d1 x d2) with SVD init from the cross-cov sum_i r_i X_i."""
    n, d1, d2 = X.shape
    M = np.einsum('n,nij->ij', r, X) / n
    U, S, Vt = svd(M, full_matrices=False)
    A = U[:, :L] * np.sqrt(S[:L] + 1e-12)
    B = Vt[:L].T * np.sqrt(S[:L] + 1e-12)
    for _ in range(iters):
        XtA = np.einsum('nij,il->njl', X, A).reshape(n, d2 * L)
        B = _ridge_solve(XtA.T @ XtA, XtA.T @ r, ridge).reshape(d2, L)
        XB = np.einsum('nij,jl->nil', X, B).reshape(n, d1 * L)
        A = _ridge_solve(XB.T @ XB, XB.T @ r, ridge).reshape(d1, L)
    return A @ B.T


def fit_rank_L_weight(X, r, L, ridge='auto', iters=30, seed=0):
    """Fit rank-L weight W (d1 x d2) so <X_i, W> ~ r_i. ALS with SVD init from the cross-covariance
    sum_i r_i X_i; `ridge` is relative to the mean Gram diagonal.

    REGIME-AWARE regularisation (ridge='auto', default): the ALS refinement OVERFITS when there is
    little data per estimated weight -- a tabular party (d2==1: the bilinear form <X, A B^T> is linear
    in the d1-vector A*b, so the alternation adds nothing but variance) or the high-dimensional regime
    d1 >= n (e.g. 384-d imaging embeddings, n~300). There we use the well-conditioned rank-L cross-
    covariance (SVD) init directly -- exactly the regularised rank-L PLS estimator, which the theory
    says the block-term reduces to on tabular/low-rank data (parity, not overfitting). Genuine multiway
    parties with enough data (d2>1 and d1<n, e.g. ECG patient x lead x time) keep the ALS. A float
    `ridge` forces the classic ALS path unchanged."""
    X = _as3(X)
    n, d1, d2 = X.shape
    L = min(L, d1, d2)
    if ridge == 'auto':
        if d2 == 1 or d1 >= n:
            iters = 0            # use the regularised cross-cov (PLS) init; skip the overfitting ALS
        ridge = 1.0
    W = _als_weight(X, r, L, ridge, iters)
    return W, np.einsum('nij,ij->n', X, W)


def _soft_threshold(W, lam):
    return np.sign(W) * np.maximum(np.abs(W) - lam, 0.0)


def fit_covert(parties, y, R=2, L=2, ridge='auto', dp_sigma=0.0, clip=None,
               sparse_lambda=0.0, seed=0, dp_clip=None, subsample=1.0,
               dp_z=0.0, dp_clip_res=None):
    """Fit COVERT. `parties` is a list of per-party feature arrays (n x ...); `y` is (n,).
    Returns a model dict. sparse_lambda>0 soft-thresholds the block weights (the sparse core).

    Differential privacy (patient-level, per-entry mechanism). Two channels cross parties each block
    and BOTH are privatised: the aggregated score t (feature leakage to the CSP) and the residual
    signal r broadcast to passive parties (label leakage). Each is clipped per PATIENT
    (|t_i|<=dp_clip, |r_i|<=dp_clip_res), so under replace-one-patient adjacency each release has L2
    sensitivity 2*clip -- O(1) in the cohort, not O(sqrt m) -- and noised with std = dp_z * (2*clip)
    (dp_z the noise multiplier; the accountant in experiments/dp_curve.py composes the 2R releases and
    reports epsilon). When subsample<1 each block fits the party factors on an independent
    Poisson(subsample) subset of patients, amplifying the per-release privacy. dp_sigma (legacy) still
    adds absolute score noise for the graceful-degradation ablation. `clip` is the old whole-vector
    party clip. The private branch activates when dp_z>0 or dp_sigma>0."""
    rng = np.random.default_rng(seed)
    Xs = [_as3(X) for X in parties]
    P = len(Xs)
    n = len(y)
    private = dp_z > 0 or dp_sigma > 0
    ymean = float(np.mean(y))
    r = np.asarray(y, float) - ymean
    Ws = [[] for _ in range(P)]
    for _ in range(R):
        # Poisson subsample of patients for this block's factor fit (privacy amplification)
        if private and subsample < 1.0:
            sub = rng.random(n) < subsample
            if sub.sum() < 2:
                sub = np.ones(n, bool)
        else:
            sub = slice(None)
        scores = []
        for p in range(P):
            W, _ = fit_rank_L_weight(Xs[p][sub], r[sub], L, ridge, seed=seed)  # local, in the clear
            if sparse_lambda > 0:
                W = _soft_threshold(W, sparse_lambda)
            s = np.einsum('nij,ij->n', Xs[p], W)             # score on all patients for the release
            if clip is not None:
                nrm = norm(s)
                if nrm > clip:
                    s = s * (clip / nrm)
            Ws[p].append(W)
            scores.append(s)
        t = np.sum(scores, axis=0)
        if dp_clip is not None:                              # score channel: per-patient clip + noise
            t = np.clip(t, -dp_clip, dp_clip)
        if dp_z > 0 and dp_clip is not None:
            t = t + rng.standard_normal(t.shape) * dp_z * (2.0 * dp_clip)
        elif dp_sigma > 0:
            t = t + rng.standard_normal(t.shape) * dp_sigma
        g = (t @ r) / (t @ t + 1e-9)
        r = r - g * t
        if dp_z > 0 and dp_clip_res is not None:             # residual channel: label-leakage privacy
            r = np.clip(r, -dp_clip_res, dp_clip_res)
            r = r + rng.standard_normal(r.shape) * dp_z * (2.0 * dp_clip_res)
    # refit all block gains jointly on train (clean) for prediction
    Ztr = _coupled_scores(Xs, Ws)
    beta = _ridge_solve(Ztr.T @ Ztr, Ztr.T @ (np.asarray(y, float) - ymean), 1e-4)
    return {'Ws': Ws, 'beta': beta, 'ymean': ymean, 'R': R, 'L': L, 'P': P}


def _coupled_scores(Xs, Ws):
    """Design matrix of R coupled block-scores: column r = sum_p <X^(p), W^(p)_r>."""
    R = len(Ws[0])
    cols = []
    for rr in range(R):
        t = np.sum([np.einsum('nij,ij->n', Xs[p], Ws[p][rr]) for p in range(len(Xs))], axis=0)
        cols.append(t)
    return np.column_stack(cols)


def predict_covert(model, parties, dp_sigma_infer=0.0, seed=1):
    """Predict. dp_sigma_infer>0 reads the prediction off the *noisy released score*
    (the Cor:dimfree inference model): adds Gaussian noise scaled to the score spread."""
    Xs = [_as3(X) for X in parties]
    Z = _coupled_scores(Xs, model['Ws'])
    if dp_sigma_infer > 0:
        rng = np.random.default_rng(seed)
        Z = Z + rng.standard_normal(Z.shape) * dp_sigma_infer * Z.std(axis=0, keepdims=True)
    return Z @ model['beta'] + model['ymean']
