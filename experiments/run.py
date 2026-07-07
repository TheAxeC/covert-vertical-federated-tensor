"""
COVERT experiment runner - main results (Table 1), ablations, and DP sweep.

Runs end-to-end NOW on synthetic stand-ins:
    python run.py --dataset synth_mimic
    python run.py --dataset synth_hcc
On the HPC with real feature files present (config.HCC_FEATURES / MIMIC_FEATURES):
    python run.py --dataset hcc
    python run.py --dataset mimic

Emits, per dataset: a comparison table (AUROC mean+/-sd over grouped-stratified folds x seeds),
the rank/party/DP ablations, and a paste-ready LaTeX \\renewcommand block for manuscript/placeholder.md.
Honest by construction: a SYNTHETIC stand-in is loudly labelled and must not be reported as real.
"""
import argparse
import warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from core import config
from core.align import verify_aligned
from core.covert import fit_covert, predict_covert
from core.baselines import (fit_p3ls, predict_p3ls, fit_centralized, fit_fbttr,
                       fit_concat, predict_concat)


def _slice(data, idx):
    return [X[idx] for X in data['parties']], data['y'][idx]


def _folds(y, groups, n_splits, seed):
    """Grouped-stratified CV split indices. Takes y+groups directly (not a data dict) so the companion
    scripts (significance/ablations_extra/dp_curve) share one fold routine with run.py."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y, groups))


def auroc(yt, sc):
    if len(np.unique(yt)) < 2:
        return np.nan
    return roc_auc_score(yt, sc)


def apr(yt, sc):
    if len(np.unique(yt)) < 2:
        return np.nan
    return average_precision_score(yt, sc)


def eval_all(data, R, L, dp_sigma, clip, sparse_lambda, n_splits, seeds, with_ap=False):
    """Return {method: (mean, sd)} AUROC across folds x seeds.
    If with_ap, also return a parallel {method: (mean, sd)} PR-AUC (average precision)."""
    methods = ['centralized', 'covert', 'p3ls', 'concat', 'single', 'fbttr']
    acc = {m: [] for m in methods}
    ap = {m: [] for m in methods}

    def _score(name, yte, sc):
        acc[name].append(auroc(yte, sc))
        if with_ap:
            ap[name].append(apr(yte, sc))

    for seed in range(seeds):
        for tr, te in _folds(data['y'], data['groups'], n_splits, seed):
            Ptr, ytr = _slice(data, tr)
            Pte, yte = _slice(data, te)
            K = R * L
            # centralized (non-private upper bound)
            m = fit_centralized(Ptr, ytr, R, L, seed=seed)
            _score('centralized', yte, predict_covert(m, Pte))
            # COVERT (private + sparse)
            m = fit_covert(Ptr, ytr, R, L, dp_sigma=dp_sigma, clip=clip,
                           sparse_lambda=sparse_lambda, seed=seed)
            _score('covert', yte, predict_covert(m, Pte, dp_sigma_infer=dp_sigma))
            # FBTTR (federated, no DP, no sparse)
            m = fit_fbttr(Ptr, ytr, R, L, seed=seed)
            _score('fbttr', yte, predict_covert(m, Pte))
            # P3LS
            m = fit_p3ls(Ptr, ytr, K=K)
            _score('p3ls', yte, predict_p3ls(m, Pte))
            # concat
            m = fit_concat(Ptr, ytr, K=K)
            _score('concat', yte, predict_concat(m, Pte))
            # best single party
            best, best_sc, best_y = -np.inf, None, None
            for p in range(len(Ptr)):
                mm = fit_p3ls([Ptr[p]], ytr, K=K)
                sc = predict_p3ls(mm, [Pte[p]])
                a = auroc(yte, sc)
                if a > best:
                    best, best_sc, best_y = a, sc, yte
            acc['single'].append(best if np.isfinite(best) else np.nan)
            if with_ap:
                ap['single'].append(apr(best_y, best_sc))
    # SD is reported SEED-LEVEL (std over the per-seed means), matching the manuscript's stated
    # "mean +/- SD over 10 seeds". The mean is unchanged; only the SD aggregation differs from a flat
    # std over all folds x seeds. acc[name] is filled in (seed, fold) order, so reshape (seeds, folds).
    def _seed_sd(v):
        v = np.asarray(v, float)
        if v.size != seeds * n_splits:
            return float(np.nanstd(v))                       # fallback if the shape is unexpected
        return float(np.nanstd(np.nanmean(v.reshape(seeds, n_splits), axis=1)))
    auroc_map = {m: (float(np.nanmean(v)), _seed_sd(v)) for m, v in acc.items()}
    if with_ap:
        ap_map = {m: (float(np.nanmean(v)), _seed_sd(v)) for m, v in ap.items()}
        return auroc_map, ap_map, acc          # acc = raw per-(seed,fold) AUROC lists (for margin CIs)
    return auroc_map


def _seed_means(v, seeds, n_splits):
    """(seeds*folds,) -> (seeds,) per-seed means (nan-safe); flat v if the shape is unexpected."""
    v = np.asarray(v, float)
    if v.size != seeds * n_splits:
        return v
    return np.nanmean(v.reshape(seeds, n_splits), axis=1)


def margin_ci(a, b, seeds, n_splits, n_boot=10000, boot_seed=0):
    """Seed-level nonparametric bootstrap 95% CI for the paired margin mean(a)-mean(b).
    a, b are the raw per-(seed,fold) AUROC lists for two methods (paired, same fold order). The margin
    is aggregated SEED-LEVEL (per-seed mean difference over the 10 seeds), matching the SD convention,
    then the 10 seed-margins are resampled with replacement. Returns (mean_margin, lo, hi)."""
    d = np.asarray(a, float) - np.asarray(b, float)          # paired per-(seed,fold) difference
    ds = _seed_means(d, seeds, n_splits)                     # -> 10 per-seed mean differences
    ds = ds[np.isfinite(ds)]
    if ds.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(boot_seed)                   # fixed seed -> reproducible CI
    boot = rng.choice(ds, size=(n_boot, ds.size), replace=True).mean(axis=1)
    return float(ds.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='synth_mimic',
                    choices=['synth_mimic', 'synth_hcc', 'hcc', 'mimic'])
    ap.add_argument('--seeds', type=int, default=config.SEEDS)
    ap.add_argument('--folds', type=int, default=config.K_FOLDS)
    ap.add_argument('--R', type=int, default=None,
                    help='deflation blocks; default = per-dataset headline (config.N_BLOCKS_BY_DATASET)')
    ap.add_argument('--L', type=int, default=config.RANK_L)
    ap.add_argument('--dp-sigma', type=float, default=config.DP_SIGMA)
    ap.add_argument('--sparse-lambda', type=float, default=0.0)
    ap.add_argument('--ablations', action='store_true')
    args = ap.parse_args()
    # Resolve the headline R for this dataset unless the user overrides it (reproduction needs no flag).
    R = args.R if args.R is not None else config.N_BLOCKS_BY_DATASET.get(args.dataset, config.N_BLOCKS)

    if args.dataset == 'synth_mimic':
        from core.data_synth import make_synth; data = make_synth(kind='mimic', n=4000)
    elif args.dataset == 'synth_hcc':
        from core.data_synth import make_synth; data = make_synth(kind='hcc', n=312)
    elif args.dataset == 'hcc':
        from core.data_hcc import load_hcc; data = load_hcc()
    else:
        from core.data_mimic import load_mimic; data = load_mimic()
    verify_aligned(data)

    n = data['y'].shape[0]
    print('=' * 74)
    print(f"COVERT pipeline | dataset={data['name']} | n={n} | parties={data['party_names']} "
          f"| prev={data['y'].mean():.3f}")
    if 'SYNTH-STANDIN' in data['name']:
        print("  *** SYNTHETIC STAND-IN - NOT REAL RESULTS (no feature file found) ***")
    print('=' * 74)

    res, res_ap, raw = eval_all(data, R, args.L, args.dp_sigma, config.CLIP_C,
                                args.sparse_lambda, args.folds, args.seeds, with_ap=True)
    print(f"\nMain comparison (mean+/-sd, R={R}, {args.folds}x{args.seeds} grouped-stratified | "
          f"prevalence={data['y'].mean():.3f}):")
    print(f"  {'method':12s}  {'AUROC':>14s}  {'PR-AUC':>14s}")
    for m in ['centralized', 'covert', 'fbttr', 'p3ls', 'concat', 'single']:
        mu, sd = res[m]; amu, asd = res_ap[m]
        print(f"  {m:12s}  {mu:.3f} +/- {sd:.3f}  {amu:.3f} +/- {asd:.3f}")
    # Reported margins with seed-level bootstrap 95% CIs (every number the manuscript quotes).
    print("\nMargins (seed-level bootstrap 95% CI over 10 seeds):")
    for lbl, hi_m, lo_m, note in [('COVERT - P3LS', 'covert', 'p3ls', 'lossless-vs-P3LS; want >=0 / CI spans 0'),
                                  ('COVERT - single', 'covert', 'single', 'fusion gain over best single party; want >0'),
                                  ('centralized - COVERT', 'centralized', 'covert', 'privacy/sparsity price; want small')]:
        m, lo, hi = margin_ci(raw[hi_m], raw[lo_m], args.seeds, args.folds)
        print(f"  > {lbl:20s} = {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  ({note})")

    # paste-ready macro block (maps to manuscript/placeholder.md)
    sfx = 'Mimic' if 'mimic' in data['name'] else ('Hcc' if 'hcc' in data['name'] else 'Synth')
    print(f"\n% paste into manuscript/placeholder.md macros ({data['name']}):")
    for m, mac in [('centralized', 'cen'), ('covert', 'vfed'), ('p3ls', 'pthree'),
                   ('concat', 'concat'), ('single', 'local')]:
        print(f"\\renewcommand{{\\{mac}{sfx}}}{{{res[m][0]:.3f}}}"
              f"  \\renewcommand{{\\{mac}{sfx}SD}}{{{res[m][1]:.3f}}}")

    # --- descriptor macros: analysed-cohort sizes + lossless gap + seed count (from data/config) ---
    # (source-cohort facts mimicPatients/mimicEcgs/mimicFs are emitted by the extractor, which sees the
    # raw record list; here we emit only what the analysed feature file + config determine.)
    y = np.asarray(data['y']); groups = np.asarray(data['groups'])
    n_pat = int(np.unique(groups).size)
    pos_pat = int(np.unique(groups[y == 1]).size)
    gap = res['centralized'][0] - res['covert'][0]
    print(f"\n% descriptor macros ({data['name']}):")
    print(f"\\renewcommand{{\\seeds}}{{{args.seeds}}}")
    if 'hcc' in data['name']:
        print(f"\\renewcommand{{\\gapHcc}}{{{gap:.1f}}}")
        print(f"\\renewcommand{{\\hccStudies}}{{{n}}}  \\renewcommand{{\\hccPatients}}{{{n}}}")
        print(f"\\renewcommand{{\\hccPts}}{{{n_pat}}}")
        print(f"\\renewcommand{{\\hccPosStudy}}{{{int(y.sum())}}}  \\renewcommand{{\\hccPosPt}}{{{pos_pat}}}")
        print(f"\\renewcommand{{\\hccVolumes}}{{{len(config.HCC_SEQUENCES)}}}  "
              f"\\renewcommand{{\\hccParties}}{{{len(data['parties'])}}}")
    elif 'mimic' in data['name']:
        ecg, labs, vitals = data['parties']
        thou = f"{n:,}".replace(',', '{,}')
        print(f"\\renewcommand{{\\gapMimic}}{{{gap:.1f}}}")
        print(f"\\renewcommand{{\\mimicCohort}}{{{thou}}}  "
              f"\\renewcommand{{\\mimicMortality}}{{{100*float(y.mean()):.1f}}}")
        print(f"\\renewcommand{{\\mimicLeads}}{{{ecg.shape[1]}}}  "
              f"\\renewcommand{{\\mimicLabs}}{{{labs.shape[1]}}}  "
              f"\\renewcommand{{\\mimicVitals}}{{{vitals.shape[1]}}}  "
              f"\\renewcommand{{\\mimicParties}}{{{len(data['parties'])}}}")

    if args.ablations:
        print("\n--- ABLATION: block rank L (the multilinear claim; L1->L2 must be +) ---")
        for L in [1, 2, 3, 4]:
            r = eval_all(data, R, L, 0.0, None, 0.0, args.folds, args.seeds)
            print(f"  L={L}: COVERT={r['covert'][0]:.3f}  P3LS={r['p3ls'][0]:.3f}")
        print("\n--- ABLATION: party count (drop passive parties) ---")
        for k in range(1, len(data['parties']) + 1):
            sub = dict(data, parties=data['parties'][:k],
                       party_names=data['party_names'][:k])
            r = eval_all(sub, R, args.L, 0.0, None, 0.0, args.folds, args.seeds)
            print(f"  parties[:{k}]={data['party_names'][:k]}: COVERT={r['covert'][0]:.3f}")
        print("\n--- ABLATION: drop one party (leave-one-out; the manuscript dropped-party row) ---")
        for j in range(len(data['parties'])):
            keep = [i for i in range(len(data['parties'])) if i != j]
            sub = dict(data, parties=[data['parties'][i] for i in keep],
                       party_names=[data['party_names'][i] for i in keep])
            r = eval_all(sub, R, args.L, 0.0, None, 0.0, args.folds, args.seeds)
            print(f"  drop {data['party_names'][j]} (keep {[data['party_names'][i] for i in keep]}): "
                  f"COVERT={r['covert'][0]:.3f}")
        print("\n--- ABLATION: DP noise sweep (graceful degradation; Cor:dimfree) ---")
        for sig in [0.0, 0.1, 0.2, 0.4, 0.8]:
            r = eval_all(data, R, args.L, sig, config.CLIP_C, 0.0, args.folds, args.seeds)
            print(f"  dp_sigma={sig:4.2f}: COVERT={r['covert'][0]:.3f}")


if __name__ == '__main__':
    warnings.filterwarnings('default')
    main()
