"""
Synthetic stand-ins matching the real datasets' SHAPES and, for imaging, their STRUCTURE, so
the pipeline runs end-to-end now and the qualitative tests (multilinear advantage, multi-party
fusion, DP) are meaningful rather than misleading.

Data contract: dict(parties=[X_p,...], party_names=[...], y=(n,), groups=(n,), name=str).

MRI realism (imaging party). A random-Gaussian tensor would misrepresent MRI. Real multi-sequence
MRI has three properties that decide whether the *multilinear* (block-term, rank>1) claim can pay
off, all modelled here EXPLICITLY (documented, not hidden):
  (1) spatial smoothness   - features are a smooth field over location (RBF-correlated), not iid;
  (2) cross-sequence corr. - the S DWI b-values are correlated views of one anatomy (shared base
                             modulated by per-sequence contrast) => the sequence mode is low-rank;
  (3) localized low-rank signal - the predictive lesion signature is a rank-L_true (sequence x
                             spatial) pattern (e.g. two distinct sequence-spatial signatures),
                             which is exactly what an L>=2 block-term can capture and a rank-1
                             (linear/PLS) model cannot.
Tabular parties (clinical/labs/vitals) stay simple correlated-linear contributors.

task='classification' -> binary label via Bernoulli(sigmoid(logit)); 'regression' -> the
continuous logit itself (where the multilinear advantage is cleanest, per the go/no-go).
"""
import numpy as np
from numpy.linalg import norm, cholesky


def _smooth_basis(F, ell, rng_dummy=None):
    grid = np.linspace(0, 1, F)
    C = np.exp(-((grid[:, None] - grid[None, :]) ** 2) / (2 * ell ** 2)) + 1e-3 * np.eye(F)
    return cholesky(C)                       # F x F: maps iid -> smooth field


def _mri_party(rng, n, S, F, L_true, scale, ell=0.12):
    """MRI-like imaging tensor (n, S, F): smooth spatial features, cross-sequence correlation,
    and a planted rank-L_true (sequence x spatial) lesion signature. Returns (X, score)."""
    Lc = _smooth_basis(F, ell)
    base = (Lc @ rng.standard_normal((F, n))).T              # (n,F) patient anatomy (smooth)
    seq_gain = 1.0 + 0.5 * rng.standard_normal(S)            # correlated per-sequence contrast
    X = np.empty((n, S, F))
    for s in range(S):
        dev = (Lc @ rng.standard_normal((F, n))).T          # sequence-specific smooth nuisance
        X[:, s, :] = seq_gain[s] * base + 0.5 * dev
    score = rng.standard_normal(n)
    W = np.zeros((S, F))                                     # rank-L_true lesion signature
    for _ in range(L_true):
        d = Lc @ rng.standard_normal(F); d /= norm(d)        # smooth spatial pattern
        c = rng.standard_normal(S); c /= norm(c)             # sequence loading
        W += np.outer(c, d)
    W /= norm(W)
    X = X + scale * np.einsum('n,sf->nsf', score, W)
    return X, score


def _tabular_party(rng, n, d, score, scale=3.0, n_corr=4):
    """Correlated-linear tabular party: features = low-rank correlated base + score along w."""
    F = rng.standard_normal((d, n_corr))
    X = (F @ rng.standard_normal((n_corr, n))).T + 0.5 * rng.standard_normal((n, d))
    w = rng.standard_normal(d); w /= norm(w)
    X = X + scale * np.outer(score, w)
    return X


def make_synth(name='synth', n=400, kind='hcc', L_true=2, snr=10.0, seed=0,
               task='classification'):
    """kind='hcc'   -> two imaging SUB-VIEW parties (decided 2026-06-18): a diffusion party
                       (DWI, 4 b-values x feat) + an anatomical party (T1W/T2W, 4 vols x feat).
                       Both MRI-like multiway tensors; no clinical/tabular party (none on disk).
       kind='mimic' -> ecg(MRI-like generator as a 12-lead x 32 multiway view) + labs + vitals."""
    rng = np.random.default_rng(seed)
    parties, names, contribs = [], [], []
    if kind == 'hcc':
        Xd, sd = _mri_party(rng, n, 4, 16, L_true, scale=3.0)   # DWI: 4 b-values
        parties += [Xd]; names += ['dwi']; contribs += [sd]
        Xa, sa = _mri_party(rng, n, 4, 16, L_true, scale=3.0)   # anatomical: T1W_IP/OOP + T2W_TEL/TES
        parties += [Xa]; names += ['anat']; contribs += [sa]
    else:
        Xe, se = _mri_party(rng, n, 12, 32, L_true, scale=3.0)   # ECG: lead x time, same structure
        parties += [Xe]; names += ['ecg']; contribs += [se]
        for nm, d in [('labs', 48), ('vitals', 9)]:
            parties += [_tabular_party(rng, n, d, (s := rng.standard_normal(n)))]
            names += [nm]; contribs += [s]
    logit = np.sum(contribs, axis=0)
    logit = (logit - logit.mean()) / (logit.std() + 1e-9)
    if task == 'regression':
        y = (logit + rng.standard_normal(n) / np.sqrt(snr)).astype(float)
    else:
        p = 1.0 / (1.0 + np.exp(-(logit + rng.standard_normal(n) / np.sqrt(snr))))
        y = (rng.random(n) < p).astype(int)
    return dict(parties=parties, party_names=names, y=y, groups=np.arange(n),
                name=f'{name}:{kind}:{task}')
