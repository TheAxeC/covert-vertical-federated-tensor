"""
COVERT ablation macros not covered by run.py's main table: party-count (partyTwo/Three), block
rank (rankOne..Four), specific dropped-party robustness (dropLabs/dropVitals), and component
stability (compStab vs a label-permutation null compStabNull). Emits a paste-ready \\renewcommand
block for the Ablations paragraph of manuscript/main.tex.

Component stability: fit COVERT across seeds on the full cohort; for each party take the first
block's flattened weight (the recovered loading), and report the mean absolute cross-seed
correlation (sign-ambiguous, so abs). The null refits on permuted labels - a stable-above-null
loading is the evidence that the recovered multiway components are inspectable substrate patterns
rather than seed noise.
"""
import argparse
import warnings
import numpy as np

from core import config
from core.align import verify_aligned
from core.covert import fit_covert, predict_covert
from experiments.run import eval_all, _folds, _slice, auroc


def _loadings(parties, y, R, L, idx):
    """First-block flattened per-party weights (the recovered loadings) for a COVERT fit on the
    resampled subset `idx` - resampling (not the inert seed) is what makes cross-fit correlation a
    genuine stability test, since the fit is otherwise deterministic given the data."""
    m = fit_covert([X[idx] for X in parties], y[idx], R, L)
    return [m['Ws'][p][0].ravel() for p in range(m['P'])]


def _per_party_abs_corr(vecs_by_fit):
    """Mean |Pearson r| across fit pairs, per party (list of length P)."""
    S = len(vecs_by_fit)
    P = len(vecs_by_fit[0])
    per_party = []
    for p in range(P):
        rs = []
        for i in range(S):
            for j in range(i + 1, S):
                a, b = vecs_by_fit[i][p], vecs_by_fit[j][p]
                if a.std() > 0 and b.std() > 0:
                    rs.append(abs(np.corrcoef(a, b)[0, 1]))
        per_party.append(float(np.mean(rs)) if rs else np.nan)
    return per_party


def component_stability(data, R, L, seeds=8, frac=0.8):
    """Per-party loading reproducibility across cohort resampling (real) vs a label-permutation null.
    Each fit is on an independent frac-subsample of the cohort; the null permutes labels INDEPENDENTLY
    per fit (so it retains no shared label structure), giving a proper floor. Returns
    (party_names, real_per_party, null_per_party) - a high real-vs-null gap on a genuinely-multiway
    party is the evidence that its recovered components carry label-driven structure, not just the
    feature covariance that both real and null inherit."""
    y = np.asarray(data['y'], float)
    n = y.shape[0]
    k = int(frac * n)
    rng = np.random.default_rng(12345)
    real = [_loadings(data['parties'], y, R, L, rng.permutation(n)[:k]) for _ in range(seeds)]
    null = [_loadings(data['parties'], rng.permutation(y), R, L, rng.permutation(n)[:k])
            for _ in range(seeds)]                       # independent label permutation per fit
    return data['party_names'], _per_party_abs_corr(real), _per_party_abs_corr(null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mimic', choices=['synth_mimic', 'synth_hcc', 'hcc', 'mimic'])
    ap.add_argument('--seeds', type=int, default=config.SEEDS)   # match run.py (10) so ablation numbers are consistent
    ap.add_argument('--folds', type=int, default=config.K_FOLDS)
    ap.add_argument('--R', type=int, default=None, help='default = per-dataset headline (HCC 2, MIMIC 8)')
    ap.add_argument('--L', type=int, default=config.RANK_L)
    ap.add_argument('--stab-seeds', type=int, default=8)
    args = ap.parse_args()
    if args.R is None:                 # match run.py: HCC overfits past R=2, MIMIC plateaus at R=8
        args.R = config.N_BLOCKS_BY_DATASET.get(args.dataset, config.N_BLOCKS)

    if args.dataset == 'synth_mimic':
        from core.data_synth import make_synth; data = make_synth(kind='mimic', n=4000)
    elif args.dataset == 'synth_hcc':
        from core.data_synth import make_synth; data = make_synth(kind='hcc', n=312)
    elif args.dataset == 'hcc':
        from core.data_hcc import load_hcc; data = load_hcc()
    else:
        from core.data_mimic import load_mimic; data = load_mimic()
    verify_aligned(data)
    names = data['party_names']
    print('=' * 74)
    print(f"COVERT extra ablations | {data['name']} | n={data['y'].shape[0]} | parties={names}")
    print('=' * 74)

    macros = {}

    # --- block rank L, full federation (flat = the multilinear DOF is not an accuracy lever;
    #     honest note: on MIMIC 2/3 parties are tabular where L>1 is inert by construction) ---
    print("\nBlock rank L (full federation, ridge='auto'):")
    for L, mac in [(1, 'rankOne'), (2, 'rankTwo'), (3, 'rankThree'), (4, 'rankFour')]:
        a = eval_all(data, args.R, L, 0.0, None, 0.0, args.folds, args.seeds)['covert'][0]
        macros[mac] = f"{a:.3f}"
        print(f"  L={L}: {a:.3f}")

    # --- honest multilinear probe: the genuinely-multiway ECG party ALONE, with the ALS refinement
    #     FORCED ON (ridge numeric, not the d2==1/d1>=n regime-switch), so L>1 is actually exercised ---
    if 'ecg' in names:
        ei = names.index('ecg')
        ecg = dict(data, parties=[data['parties'][ei]], party_names=['ecg'])
        print("\nBlock rank L (ECG party only, ALS forced on, ridge=1.0):")
        for L, mac in [(1, 'rankEcgOne'), (2, None), (3, 'rankEcgThree')]:
            accs = []
            for seed in range(args.seeds):
                for tr, te in _folds(ecg['y'], ecg['groups'], args.folds, seed):
                    Ptr, ytr = _slice(ecg, tr); Pte, yte = _slice(ecg, te)
                    m = fit_covert(Ptr, ytr, args.R, L, ridge=1.0, seed=seed)
                    accs.append(auroc(yte, predict_covert(m, Pte)))
            a = float(np.nanmean(accs))
            print(f"  L={L}: {a:.3f}")
            if mac:
                macros[mac] = f"{a:.3f}"

    # --- party count: 2 parties (first two) then all (adding views helps then saturates) ---
    print("\nParty count (cumulative):")
    for k, mac in [(2, 'partyTwo'), (len(data['parties']), 'partyThree')]:
        sub = dict(data, parties=data['parties'][:k], party_names=names[:k])
        a = eval_all(sub, args.R, args.L, 0.0, None, 0.0, args.folds, args.seeds)['covert'][0]
        macros[mac] = f"{a:.3f}"
        print(f"  parties[:{k}]={names[:k]}: {a:.3f}")

    # --- dropped-party robustness (drop each passive party by name if present) ---
    print("\nDropped passive party:")
    for drop_name, mac in [('labs', 'dropLabs'), ('vitals', 'dropVitals')]:
        if drop_name in names:
            keep = [i for i, nm in enumerate(names) if nm != drop_name]
            sub = dict(data, parties=[data['parties'][i] for i in keep],
                       party_names=[names[i] for i in keep])
            a = eval_all(sub, args.R, args.L, 0.0, None, 0.0, args.folds, args.seeds)['covert'][0]
            macros[mac] = f"{a:.3f}"
            print(f"  drop {drop_name} -> {[names[i] for i in keep]}: {a:.3f}")

    # --- component stability vs label-permutation null, PER PARTY ---
    pnames, real_pp, null_pp = component_stability(data, args.R, args.L, seeds=args.stab_seeds)
    print("\nComponent reproducibility (per party, real vs label-permuted null):")
    for nm, rr, nn in zip(pnames, real_pp, null_pp):
        print(f"  {nm:8s}: real {rr:.2f}  null {nn:.2f}  gap {rr-nn:+.2f}")
    # macros: overall mean + the genuinely-multiway ECG party specifically
    macros['compStab'] = f"{np.nanmean(real_pp):.2f}"
    macros['compStabNull'] = f"{np.nanmean(null_pp):.2f}"
    if 'ecg' in pnames:
        ei = pnames.index('ecg')
        macros['compStabEcg'] = f"{real_pp[ei]:.2f}"
        macros['compStabEcgNull'] = f"{null_pp[ei]:.2f}"

    print(f"\n% paste into manuscript/main.tex ablation macros ({data['name']}):")
    for mac, val in macros.items():
        print(f"\\renewcommand{{\\{mac}}}{{{val}}}")


if __name__ == '__main__':
    warnings.filterwarnings('default')
    main()
