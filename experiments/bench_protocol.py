"""
COVERT protocol-cost benchmark - the measured (not estimated) numbers behind the manuscript's
Route A vs Route B cost table, the convergence/wall-clock/PSI figures, and the communication
scalability ablation. Companion to run.py (which owns the accuracy macros); this owns the
protocol/cost macros.

What it measures on the real aligned cohort (or a synthetic stand-in, loudly labelled):
  * Communication per round, from the ACTUAL party tensor shapes (float64, 8 bytes):
      - Route B (secure agg.): each party sends its length-m partial score, the CSP broadcasts
        the summed score, the active party broadcasts the m x C residual -> m*(P + 1 + C) floats.
        Independent of the feature-mode dimensions (the headline economy).
      - Route A (masked HOOI): the parties send masked full tensors so the CSP can run dense HOOI
        -> sum_p prod(shape of party p) = sum_p n*prod(I_n^p) floats. Grows with the feature size.
    Both are message-size counts, so the figures are measured rather than modelled.
  * Core sparsity: Route B keeps the sparse block core (fraction of exactly-zero weight entries at
    the largest sparse_lambda that preserves accuracy); Route A forces a dense core (0% zeros).
  * Route A vs Route B AUROC at equal expressiveness (dense lambda=0 vs sparse lambda*).
  * Convergence: AUROC vs number of deflation blocks R (the empirical face of the exactness theorem).
  * Wall-clock: one federated Route-B fit vs one centralized fit.
  * Entity alignment (PSI): wall-time of the set intersection over the cohort identifiers.
  * Scalability: Route B comm stays flat as the feature dimension grows; Route A scales with it.

Emits a paste-ready \\renewcommand block for the protocol/cost macros in manuscript/main.tex.
"""
import argparse
import time
import warnings
import numpy as np

from core import config
from core.align import verify_aligned, private_set_intersection
from core.covert import fit_covert, predict_covert
from core.baselines import fit_centralized
from experiments.run import eval_all, _folds, _slice, auroc

_BYTES = 8               # float64 payloads
_MB = 1e6


def _party_shapes(data):
    """(n, C, [feature-size per party]) from the real aligned tensors."""
    n = data['y'].shape[0]
    C = 1                                     # single active-party label column
    fdims = [int(np.prod(X.shape[1:])) for X in data['parties']]   # prod(I_n^p), no sample axis
    return n, C, fdims


def comm_per_round_mb(n, C, fdims, feat_scale=1.0):
    """Per-round communication (MB) for both routes at a given feature-dimension scale.
    Route B = m*(P + 1 + C) floats (score up, summed score down, residual); feature-independent.
    Route A = sum_p n*prod(I_n^p) floats (masked full tensors for dense HOOI); feature-scaled."""
    P = len(fdims)
    route_b = n * (P + 1 + C)                                   # cohort-sized, flat in features
    route_a = sum(n * fd * feat_scale for fd in fdims) + n * C  # data-sized, grows with features
    return route_a * _BYTES / _MB, route_b * _BYTES / _MB


def core_sparsity(parties, y, R, L, lam, seed=0):
    """Fraction of exactly-zero block-weight entries under the sparse (Route B) fit."""
    m = fit_covert(parties, y, R, L, sparse_lambda=lam, seed=seed)
    zeros = total = 0
    for p in range(m['P']):
        for W in m['Ws'][p]:
            zeros += int(np.sum(W == 0.0))
            total += int(W.size)
    return zeros / max(total, 1)


def pick_sparse_lambda(data, R, L, folds, seeds, tol=0.005):
    """Largest sparse_lambda whose COVERT AUROC stays within tol of the dense (lambda=0) fit -
    the operating point where Route B keeps the sparse core at no accuracy cost."""
    dense = eval_all(data, R, L, 0.0, None, 0.0, folds, seeds)['covert'][0]
    best_lam, best_auroc = 0.0, dense
    for lam in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        a = eval_all(data, R, L, 0.0, None, lam, folds, seeds)['covert'][0]
        if a >= dense - tol:
            best_lam, best_auroc = lam, a          # still accurate: accept the sparser core
        else:
            break                                   # past the knee; stop
    return best_lam, dense, best_auroc


def time_fits(data, R, L, seed=0):
    """Wall-clock (seconds) of one federated Route-B fit vs one centralized fit on the full cohort."""
    P, y = data['parties'], data['y']
    t0 = time.perf_counter(); fit_covert(P, y, R, L, seed=seed); t_fed = time.perf_counter() - t0
    t0 = time.perf_counter(); fit_centralized(P, y, R, L, seed=seed); t_cen = time.perf_counter() - t0
    return t_fed, t_cen


def time_psi(data, seed=0):
    """Wall-clock (seconds) of the entity-alignment set intersection over the cohort identifiers.
    Plaintext reference (align.py); the deployed ECDH-PSI yields the same aligned cohort."""
    n = data['y'].shape[0]
    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    parties = []
    for _ in range(len(data['parties'])):
        extra = rng.integers(n, 2 * n, size=n // 4)          # non-overlapping ids per party
        parties.append(np.concatenate([ids, extra]))
    t0 = time.perf_counter()
    private_set_intersection(parties)
    return time.perf_counter() - t0


def convergence_curve(data, L, folds, seeds, Rs=(1, 2, 4, 6, 8)):
    """COVERT AUROC vs number of deflation blocks R; roundsConv = smallest R within 0.005 of the max."""
    curve = [(R, eval_all(data, R, L, 0.0, None, 0.0, folds, seeds)['covert'][0]) for R in Rs]
    top = max(a for _, a in curve)
    conv = next((R for R, a in curve if a >= top - 0.005), curve[-1][0])
    return curve, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mimic', choices=['synth_mimic', 'synth_hcc', 'hcc', 'mimic'])
    ap.add_argument('--seeds', type=int, default=config.SEEDS)   # match run.py (10) so accuracy points are consistent
    ap.add_argument('--folds', type=int, default=config.K_FOLDS)
    ap.add_argument('--R', type=int, default=None)               # default = per-dataset headline (HCC 2, MIMIC 8)
    ap.add_argument('--L', type=int, default=config.RANK_L)
    ap.add_argument('--feat-large', type=float, default=8.0)     # feature-scale for the scalability sweep
    ap.add_argument('--quick', action='store_true', help='skip the accuracy sweeps (comm/wall/PSI only)')
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

    n, C, fdims = _party_shapes(data)
    P = len(fdims)
    print('=' * 74)
    print(f"COVERT protocol benchmark | dataset={data['name']} | n={n} | P={P} "
          f"| feat-dims={fdims} (sum={sum(fdims)})")
    if 'SYNTH-STANDIN' in data['name']:
        print("  *** SYNTHETIC STAND-IN - NOT REAL RESULTS (no feature file found) ***")
    print('=' * 74)

    # --- communication (from real shapes) ---
    comm_a, comm_b = comm_per_round_mb(n, C, fdims)
    comm_a_lg, comm_b_lg = comm_per_round_mb(n, C, fdims, feat_scale=args.feat_large)
    ratio = comm_a / comm_b if comm_b else float('inf')
    print(f"\nCommunication per round (float64):")
    print(f"  Route A (masked full tensors) = {comm_a:.2f} MB")
    print(f"  Route B (secure-agg score)    = {comm_b:.3f} MB   (Route A / Route B = {ratio:.0f}x)")
    print(f"  scalability x{args.feat_large:g} feature dim: "
          f"Route A {comm_a:.2f}->{comm_a_lg:.2f} MB (grows), "
          f"Route B {comm_b:.3f}->{comm_b_lg:.3f} MB (flat)")

    # --- PSI + wall-clock ---
    psi_s = time_psi(data)
    t_fed, t_cen = time_fits(data, args.R, args.L)
    print(f"\nEntity alignment (PSI): {psi_s*1e3:.1f} ms")
    print(f"Wall-clock (one fit): federated {t_fed:.2f} s | centralized {t_cen:.2f} s")

    macros = {
        'commRouteA': f"{comm_a:.1f}", 'commRouteB': f"{comm_b:.2f}",
        'commRatio': f"{ratio:.0f}",
        'routeaSparsity': "0",
        'commSmall': f"{comm_b:.2f}", 'commLarge': f"{comm_b_lg:.2f}",
        'commSmallA': f"{comm_a:.1f}", 'commLargeA': f"{comm_a_lg:.1f}",
        'psiMillis': f"{psi_s*1e3:.0f}",              # one-time entity alignment (ms, plaintext reference)
        'wallVfed': f"{t_fed:.1f}", 'wallCen': f"{t_cen:.1f}",
    }

    if not args.quick:
        # --- sparsity + Route A/B accuracy at the sparse operating point ---
        lam, dense_auroc, sparse_auroc = pick_sparse_lambda(data, args.R, args.L, args.folds, args.seeds)
        spars = core_sparsity(data['parties'], data['y'], args.R, args.L, lam) if lam > 0 else 0.0
        print(f"\nSparse operating point: lambda*={lam:g} "
              f"| dense (Route A) AUROC {dense_auroc:.3f} | sparse (Route B) AUROC {sparse_auroc:.3f} "
              f"| core sparsity {spars*100:.0f}% zeros")
        # --- convergence ---
        curve, conv = convergence_curve(data, args.L, args.folds, args.seeds)
        print("Convergence (AUROC vs R): " + "  ".join(f"R{R}={a:.3f}" for R, a in curve)
              + f"  -> roundsConv={conv}")
        macros.update({
            'routeaMimic': f"{dense_auroc:.3f}", 'routebMimic': f"{sparse_auroc:.3f}",
            'routebSparsity': f"{spars*100:.0f}", 'roundsConv': f"{conv}",
        })

    sfx = 'Mimic' if 'mimic' in data['name'] else ('Hcc' if 'hcc' in data['name'] else 'Synth')
    print(f"\n% paste into manuscript/main.tex protocol/cost macros ({data['name']}):")
    for mac, val in macros.items():
        print(f"\\renewcommand{{\\{mac}}}{{{val}}}")


if __name__ == '__main__':
    warnings.filterwarnings('default')
    main()
