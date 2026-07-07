"""
Differential-privacy utility curve for COVERT: AUROC vs the ACCOUNTED patient-level privacy budget
epsilon, so the paper's DP contribution is costed, not only proven.

Mechanism (core/covert.py, dp_sigma>0): each of the R deflation blocks releases the aggregated
per-patient coupling score with per-PATIENT clipping (|t_i| <= dp_clip) and added Gaussian noise.
Under replace-one-patient adjacency the released vector changes in essentially one entry, so the L2
sensitivity is 2*dp_clip -- O(1) in the cohort, NOT the O(sqrt m) of a whole-vector party clip -- which
is what makes small-epsilon privacy usable. When subsample<1, each block fits the party factors on an
independent Poisson(subsample) subset of patients, so the subsampled-Gaussian mechanism amplifies the
per-release privacy. We account the R releases with the subsampled-Gaussian Renyi-DP bound
(Mironov et al. 2019) and convert RDP -> (eps, delta). The residual broadcast is released under the same
2*dp_clip-sensitivity mechanism (constant-factor more releases, not more sigma), so the dimension-free-
in-feature-dimension property is unaffected; we account the score releases and note this in the paper.
"""
import argparse
import warnings
import numpy as np
from scipy.special import logsumexp, gammaln

from core import config
from core.align import verify_aligned
from core.covert import fit_covert, predict_covert, _as3, fit_rank_L_weight
from experiments.run import _folds, _slice, auroc


def _logcomb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def subsampled_gaussian_rdp(q, z, orders=range(2, 65)):
    """RDP epsilon(order) of one Poisson-subsampled Gaussian release, rate q, noise multiplier
    z = sigma / sensitivity. Mironov-Talwar-Zhang integer-order upper bound. Returns {order: rdp}."""
    out = {}
    for a in orders:
        if q >= 1.0:
            out[a] = a / (2.0 * z ** 2)
            continue
        logterms = [_logcomb(a, k) + (a - k) * np.log(1 - q) + k * np.log(q)
                    + (k * (k - 1)) / (2.0 * z ** 2) for k in range(a + 1)]
        out[a] = logsumexp(logterms) / (a - 1.0)
    return out


def accounted_eps(q, z, releases, delta, orders=range(2, 65)):
    """Total (eps, delta) after `releases` subsampled-Gaussian releases at noise multiplier
    z = sigma/sensitivity, minimised over Renyi order."""
    if z <= 0:
        return float('inf')
    rdp = subsampled_gaussian_rdp(q, z, orders)
    return float(min(releases * rdp[a] + np.log(1.0 / delta) / (a - 1.0) for a in orders))


def patient_clip(data, R, L, pct=95):
    """Per-patient score-entry clip c_pat: the `pct` percentile of |t_i| (mild clipping at the op point)."""
    Xs = [_as3(X) for X in data['parties']]
    r = np.asarray(data['y'], float) - np.mean(data['y'])
    t = np.zeros(len(r))
    for X in Xs:
        _, s = fit_rank_L_weight(X, r, L)
        t = t + s
    return float(np.percentile(np.abs(t), pct))


def covert_auroc(data, R, L, dp_z, dp_clip, dp_clip_res, subsample, folds, seeds):
    """Mean CV AUROC of COVERT under the two-channel (score + residual) patient-level DP mechanism."""
    accs = []
    for seed in range(seeds):
        for tr, te in _folds(data['y'], data['groups'], folds, seed):
            Ptr, ytr = _slice(data, tr); Pte, yte = _slice(data, te)
            m = fit_covert(Ptr, ytr, R, L, dp_z=dp_z, dp_clip=dp_clip, dp_clip_res=dp_clip_res,
                           subsample=subsample, seed=seed)
            accs.append(auroc(yte, predict_covert(m, Pte)))
    return float(np.nanmean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mimic', choices=['synth_mimic', 'synth_hcc', 'hcc', 'mimic'])
    ap.add_argument('--seeds', type=int, default=config.SEEDS)   # match run.py (10) so DP-curve points are consistent
    ap.add_argument('--folds', type=int, default=config.K_FOLDS)
    ap.add_argument('--R', type=int, default=None, help='default = per-dataset headline (HCC 2, MIMIC 8)')
    ap.add_argument('--L', type=int, default=config.RANK_L)
    ap.add_argument('--delta', type=float, default=1e-5)
    ap.add_argument('--subsample', type=float, default=0.1, help='Poisson rate q (amplification)')
    args = ap.parse_args()
    if args.R is None:                 # match run.py: HCC overfits past R=2, MIMIC plateaus at R=8
        args.R = config.N_BLOCKS_BY_DATASET.get(args.dataset, config.N_BLOCKS)

    if args.dataset == 'hcc':
        from core.data_hcc import load_hcc; data = load_hcc()
    elif args.dataset == 'mimic':
        from core.data_mimic import load_mimic; data = load_mimic()
    elif args.dataset == 'synth_hcc':
        from core.data_synth import make_synth; data = make_synth(kind='hcc', n=312)
    else:
        from core.data_synth import make_synth; data = make_synth(kind='mimic', n=4000)
    verify_aligned(data)

    c_t = patient_clip(data, args.R, args.L)                  # score-channel per-patient clip
    r0 = np.asarray(data['y'], float) - np.mean(data['y'])
    c_r = float(np.percentile(np.abs(r0), 100))               # residual-channel clip (label-centred)
    releases = 2 * args.R                                     # score + residual, per block
    n = data['y'].shape[0]
    print(f"{data['name']} | n={n} | {releases} releases (score+residual) | delta={args.delta} "
          f"| q={args.subsample} | clips: score {c_t:.3g}, residual {c_r:.3g}")

    a0 = covert_auroc(data, args.R, args.L, 0.0, None, None, 1.0, args.folds, args.seeds)
    print(f"  eps=inf (no noise):  AUROC {a0:.3f}")
    rows = [('inf', a0)]
    # score-only release (feature leakage to the CSP; R releases, residual left clean) -- the free point
    z_free = 8.0
    eps_free = accounted_eps(args.subsample, z_free, args.R, args.delta)
    a_free = covert_auroc(data, args.R, args.L, z_free, c_t, None, args.subsample, args.folds, args.seeds)
    print(f"  eps={eps_free:7.2f} (z={z_free}, SCORE-ONLY):  AUROC {a_free:.3f}")
    zres = {}
    for z in (0.5, 1.0, 2.0, 4.0, 8.0):                       # noise multiplier sigma/sensitivity
        eps = accounted_eps(args.subsample, z, releases, args.delta)
        a = covert_auroc(data, args.R, args.L, z, c_t, c_r, args.subsample, args.folds, args.seeds)
        print(f"  eps={eps:7.2f} (z={z}):  AUROC {a:.3f}")
        rows.append((f"{eps:.2f}", a)); zres[z] = (eps, a)

    print("\n% DP utility curve (accounted patient-level eps, AUROC), both channels privatised:")
    print("  " + "  ".join(f"({e},{a:.3f})" for e, a in rows))

    if 'mimic' in data['name']:            # the manuscript DP macros: free (score-only), moderate, strong
        eps_mod, a_mod = zres[1.0]; _, a_strong = zres[8.0]
        print("\n% DP macros (MIMIC):")
        print(f"\\renewcommand{{\\dpEpsFree}}{{{eps_free:.2f}}}  \\renewcommand{{\\dpAurocFree}}{{{a_free:.3f}}}")
        print(f"\\renewcommand{{\\dpEpsMod}}{{{eps_mod:.1f}}}  \\renewcommand{{\\dpAurocMod}}{{{a_mod:.3f}}}")
        print(f"\\renewcommand{{\\dpAurocStrong}}{{{a_strong:.3f}}}")


if __name__ == '__main__':
    warnings.filterwarnings('default')
    main()
