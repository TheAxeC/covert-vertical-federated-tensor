"""
Paired significance for the manuscript's parity + margin claims. Two layers:
  * DIFFERENCE test - COVERT vs each baseline across the seed x fold AUROCs, paired Wilcoxon
    signed-rank with Holm correction (backs the margin claims like "beats best single party").
  * EQUIVALENCE test - a difference test that fails to reject is NOT evidence of parity, so for the
    parity claim we aggregate to SEED-LEVEL independent units (one mean AUROC per seed) and run TOST
    (two one-sided tests) against a pre-specified margin, plus a 95% CI on the paired difference.
    This is the correct instrument for "statistically indistinguishable from P3LS/centralized".
"""
import argparse
import warnings
import numpy as np
from scipy.stats import wilcoxon, t as student_t

from core import config
from core.align import verify_aligned
from core.covert import fit_covert, predict_covert
from core.baselines import (fit_p3ls, predict_p3ls, fit_centralized, fit_concat, predict_concat)
from experiments.run import _folds, _slice, auroc


def paired_aurocs(data, R, L, folds, seeds):
    """Per-(seed,fold) AUROC for covert / centralized / p3ls / concat / single, plus a parallel
    seed-index array so callers can aggregate to seed-level independent units."""
    rows = {m: [] for m in ('covert', 'centralized', 'p3ls', 'concat', 'single')}
    seed_ix = []
    for seed in range(seeds):
        for tr, te in _folds(data['y'], data['groups'], folds, seed):
            Ptr, ytr = _slice(data, tr); Pte, yte = _slice(data, te)
            K = R * L
            m = fit_covert(Ptr, ytr, R, L, seed=seed)
            rows['covert'].append(auroc(yte, predict_covert(m, Pte)))
            m = fit_centralized(Ptr, ytr, R, L, seed=seed)
            rows['centralized'].append(auroc(yte, predict_covert(m, Pte)))
            m = fit_p3ls(Ptr, ytr, K=K)
            rows['p3ls'].append(auroc(yte, predict_p3ls(m, Pte)))
            m = fit_concat(Ptr, ytr, K=K)
            rows['concat'].append(auroc(yte, predict_concat(m, Pte)))
            best = -np.inf
            for p in range(len(Ptr)):
                mm = fit_p3ls([Ptr[p]], ytr, K=K)
                best = max(best, auroc(yte, predict_p3ls(mm, [Pte[p]])))
            rows['single'].append(best)
            seed_ix.append(seed)
    out = {m: np.array(v) for m, v in rows.items()}
    out['_seed'] = np.array(seed_ix)
    return out


def _seed_means(vals, seed_ix):
    """Aggregate per-(seed,fold) values to one mean per seed (independent units), NaN-safe: a degenerate
    single-class test fold scores NaN (small cohorts like HCC), so average only the finite folds."""
    out = []
    for s in np.unique(seed_ix):
        v = vals[seed_ix == s]; v = v[np.isfinite(v)]
        out.append(v.mean() if v.size else np.nan)
    return np.array(out)


def tost_equivalence(cov_seed, base_seed, margin):
    """TOST on paired seed-level differences: equivalent within +/-margin if the 90% CI of the mean
    paired difference (for alpha=0.05 two-one-sided) lies inside (-margin, margin). Returns
    (mean_diff, lo95, hi95, equivalent_bool, tost_p)."""
    d = cov_seed - base_seed
    n = len(d)
    md, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n) if sd > 0 else 1e-12
    # 95% CI (report) and 90% CI (TOST decision at alpha=0.05)
    tcrit95 = student_t.ppf(0.975, n - 1)
    tcrit90 = student_t.ppf(0.95, n - 1)
    lo95, hi95 = md - tcrit95 * se, md + tcrit95 * se
    lo90, hi90 = md - tcrit90 * se, md + tcrit90 * se
    equivalent = (lo90 > -margin) and (hi90 < margin)
    # TOST p-value: max of the two one-sided p-values
    t_lower = (md + margin) / se        # H0: diff <= -margin
    t_upper = (margin - md) / se        # H0: diff >= +margin
    p_lower = 1 - student_t.cdf(t_lower, n - 1)
    p_upper = 1 - student_t.cdf(t_upper, n - 1)
    return md, lo95, hi95, equivalent, max(p_lower, p_upper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mimic', choices=['synth_mimic', 'synth_hcc', 'hcc', 'mimic'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--folds', type=int, default=config.K_FOLDS)
    ap.add_argument('--R', type=int, default=None, help='default = per-dataset headline (HCC 2, MIMIC 8)')
    ap.add_argument('--L', type=int, default=config.RANK_L)
    ap.add_argument('--margin', type=float, default=0.01, help='TOST equivalence margin (AUROC)')
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

    rows = paired_aurocs(data, args.R, args.L, args.folds, args.seeds)
    cov = rows['covert']
    print(f"{data['name']} | n={data['y'].shape[0]} | paired over {len(cov)} (seed x fold) AUROCs")
    print(f"  COVERT mean {np.nanmean(cov):.3f}")
    baselines = ['p3ls', 'concat', 'single']
    pvals = {}
    for b in baselines:
        d = cov - rows[b]
        d = d[np.isfinite(d)]                         # NaN-safe: drop degenerate single-class folds (HCC)
        if d.size == 0 or np.all(d == 0):
            p = 1.0                                   # no signal / identical -> not distinguishable
        else:
            try:
                _, p = wilcoxon(d)
            except ValueError:
                p = 1.0
        pvals[b] = p
    # Holm correction over the three baselines
    order = sorted(baselines, key=lambda b: pvals[b])
    holm = {}
    for i, b in enumerate(order):
        holm[b] = min(1.0, pvals[b] * (len(order) - i))
    for b in baselines:
        d = np.nanmean(cov - rows[b])
        verdict = 'indistinguishable' if holm[b] > 0.05 else 'significant'
        print(f"  COVERT - {b:7s} = {d:+.3f}  Wilcoxon p={pvals[b]:.3g}  Holm p={holm[b]:.3g}  ({verdict})")

    # --- EQUIVALENCE (parity) at seed level: TOST + 95% CI against a pre-specified margin ---
    seed_ix = rows['_seed']
    cov_s = _seed_means(cov, seed_ix)
    print(f"\n  Equivalence (parity), seed-level n={len(cov_s)} independent units, margin +/-{args.margin}:")
    for b in ('p3ls', 'centralized'):
        base_s = _seed_means(rows[b], seed_ix)
        mask = np.isfinite(cov_s) & np.isfinite(base_s)          # pair only seeds finite in both
        md, lo, hi, equiv, ptost = tost_equivalence(cov_s[mask], base_s[mask], args.margin)
        tag = 'EQUIVALENT' if equiv else 'not shown equivalent'
        print(f"    COVERT - {b:11s} = {md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"TOST p={ptost:.3g}  ({tag} at +/-{args.margin})")
        # the manuscript's TOST parity macros are the powered-arm (MIMIC) COVERT-vs-P3LS equivalence
        if b == 'p3ls' and 'mimic' in data['name']:
            print(f"\\renewcommand{{\\tostDiff}}{{{md:+.4f}}}  \\renewcommand{{\\tostLo}}{{{lo:+.4f}}}  "
                  f"\\renewcommand{{\\tostHi}}{{{hi:+.4f}}}  \\renewcommand{{\\tostMargin}}{{{args.margin:g}}}")


if __name__ == '__main__':
    warnings.filterwarnings('default')
    main()
