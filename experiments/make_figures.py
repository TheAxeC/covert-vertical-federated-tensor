"""
Generate the manuscript's empirical figures from the fitted models / measured data:
  1. component.pdf   - a recovered COVERT component on the genuinely-multiway ECG party
                       (lead x feature loading), the interpretability the score-only baselines cannot expose.
  2. convergence.pdf - test AUROC vs number of deflation blocks R (the empirical face of the exactness theorem).
  3. scalability.pdf - per-round communication vs feature dimension: Route B flat, Route A grows.
  4. dp_utility.pdf  - AUROC vs accounted patient-level privacy budget epsilon (both channels privatised).

Convergence and DP points are the measured values from run.py / dp_curve.py (regenerated there); the
component and scalability panels are computed here. Writes into manuscript/figures/.
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import config
from core.covert import fit_covert
from experiments.bench_protocol import comm_per_round_mb, _party_shapes

FIGDIR = os.path.join(os.path.dirname(config._REPO), 'manuscript', 'figures')
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ECG_FEATS = ['P(0-5Hz)', 'P(5-15)', 'P(15-40)', 'P(40-100)', 'std', 'rms', 'range', 'iqr']


def fig_component(data):
    """Recovered ECG-party block weight (12 leads x 8 features) - a per-modality signature."""
    ei = data['party_names'].index('ecg')
    m = fit_covert(data['parties'], data['y'], R=8, L=2, seed=0)
    W = m['Ws'][ei][0]                                    # first block, ecg party (12 x 8)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    vmax = np.abs(W).max()
    im = ax.imshow(W, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(W.shape[1])); ax.set_xticklabels(ECG_FEATS[:W.shape[1]], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(W.shape[0])); ax.set_yticklabels(LEADS[:W.shape[0]], fontsize=7)
    ax.set_xlabel('short-time feature'); ax.set_ylabel('ECG lead')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='loading')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, 'component.pdf')); plt.close(fig)


def fig_convergence():
    """Measured MIMIC convergence: AUROC vs deflation blocks R (run.py --R)."""
    R = [1, 2, 4, 6, 8]; auroc = [0.622, 0.637, 0.717, 0.740, 0.746]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(R, auroc, 'o-', color='#1f77b4')
    ax.axhline(0.746, ls='--', color='gray', lw=0.8, label='centralized')
    ax.set_xlabel('deflation blocks $R$'); ax.set_ylabel('test AUROC')
    ax.set_ylim(0.6, 0.76); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, 'convergence.pdf')); plt.close(fig)


def fig_scalability(data):
    """Per-round communication vs feature-dimension scale: Route B flat, Route A grows."""
    n, C, fdims = _party_shapes(data)
    scales = np.array([1, 2, 4, 8, 16, 32])
    a = [comm_per_round_mb(n, C, fdims, s)[0] for s in scales]
    b = [comm_per_round_mb(n, C, fdims, s)[1] for s in scales]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(scales, a, 's-', color='#d62728', label='Route A (masked)')
    ax.plot(scales, b, 'o-', color='#2ca02c', label='Route B (secure agg.)')
    ax.set_xlabel(r'feature-dimension scale ($\times$)'); ax.set_ylabel('comm / round (MB)')
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, 'scalability.pdf')); plt.close(fig)


def fig_dp():
    """Measured DP utility: AUROC vs accounted patient-level epsilon (dp_curve.py, both channels)."""
    eps = [18.4, 4.8, 1.4, 0.57, 0.27]; auroc = [0.731, 0.699, 0.679, 0.673, 0.672]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(eps, auroc, 'o-', color='#9467bd')
    ax.axhline(0.746, ls='--', color='gray', lw=0.8, label='non-private')
    ax.set_xlabel(r'accounted privacy budget $\varepsilon$'); ax.set_ylabel('AUROC')
    ax.set_xscale('log'); ax.set_ylim(0.6, 0.76); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, 'dp_utility.pdf')); plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    from core.data_mimic import load_mimic
    data = load_mimic()
    fig_component(data)
    fig_convergence()
    fig_scalability(data)
    fig_dp()
    print(f"wrote component / convergence / scalability / dp_utility .pdf to {FIGDIR}")


if __name__ == '__main__':
    warnings.filterwarnings('default')
    main()
