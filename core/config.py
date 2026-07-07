"""
COVERT real-data pipeline - configuration.

Path conventions mirror HCCNet (publications/[R] HCCNet_Revision/code/utils/config.py; HCCstudy dissolved 2026-06-19): every path is
env-overridable so the common case needs no flags, and defaults point at the UTwente shares.

The pipeline runs end-to-end NOW on synthetic stand-ins (--dataset synth). Real datasets need:
  * HCCNet: precomputed per-sequence imaging features + a clinical table (see data_hcc.py).
            DATA dir defaults to the deepstore share used by HCCNet.
  * MIMIC-IV-ECG: credentialed PhysioNet download + linkage (see data_mimic.py).
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../code

# --- data locations (env-overridable, HCCNet-style) ------------------------------------
HCC_DATA_DIR = os.environ.get('HCCNET_DATA_DIR', '/deepstore/datasets/bms/hcc_study')
# COVERT consumes FEATURES, not raw voxels: a (patients x sequences x feat) imaging tensor
# (e.g. HCCNet pretrained-backbone embeddings) + a clinical table. Produce with HCCNet's
# backbone (see data_hcc.py header); point this at the resulting .npz.
HCC_FEATURES = os.environ.get('COVERT_HCC_FEATURES',
                              os.path.expanduser('~/projects/covert/hcc_embeddings.npz'))

MIMIC_DATA_DIR = os.environ.get('COVERT_MIMIC_DIR', '/deepstore/datasets/mimic-iv-ecg')
MIMIC_FEATURES = os.environ.get('COVERT_MIMIC_FEATURES',
                                os.path.expanduser('~/data/mimic-iv/covert/mimic_features.npz'))

RESULTS_DIR = os.environ.get('COVERT_RESULTS_DIR', os.path.join(_REPO, 'results'))

# --- HCCNet imaging modalities -----------------------------------------------------------
# MULTI-SEQUENCE (decided 2026-06-18): the real cohort on deepstore carries genuinely distinct
# MRI sequences per study, not just DWI. Imaging party now spans DWI (4 b-values) + T1-weighted
# (in/out-of-phase) + T2-weighted (two echo times) = 8 co-registered volumes. (Further T1 dynamic
# phases T1A/T1V/T1D + T1WI/T1W_QNT also exist on disk; left out for now.)
HCC_SEQUENCES = ['DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800',
                 'T1W_IP', 'T1W_OOP', 'T2W_TEL', 'T2W_TES']
# Vertical SUB-VIEW split (decided 2026-06-18): no clinical table on disk -> two imaging parties.
HCC_PARTIES = {'dwi':  ['DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800'],     # diffusion party
               'anat': ['T1W_IP', 'T1W_OOP', 'T2W_TEL', 'T2W_TES']}       # anatomical party
# Label = incident-HCC within a horizon (labels_<H>months.csv). 12mo default: the analysed leak-free
# OOF cache is 400 studies from 163 patients, 42 study-level pos (10.5%) / 17 patient-level pos.
HCC_LABEL_HORIZON = int(os.environ.get('COVERT_HCC_HORIZON', '12'))        # one of 3/6/12/24/36
# --- MIMIC parties (decided 2026-06-18: 3-party ECG + labs + OMR-vitals) -----------------
MIMIC_PARTIES = ['ecg', 'labs', 'vitals']            # ecg = 3-way tensor; labs/vitals tabular (OMR vitals)

# --- model / protocol defaults (align with manuscript/placeholder.md) -------------------
N_BLOCKS = 2          # R deflation blocks (fallback; headline R is per-dataset below)
# Headline deflation-block count R is DATASET-DEPENDENT and baked in so the reproduction needs
# NO manual --R flag: MIMIC converges/plateaus at R=8; HCC (d>>n) overfits past R=2, so R=2 is its
# operating point. run.py resolves R from here when --R is not given explicitly.
N_BLOCKS_BY_DATASET = {'mimic': 8, 'synth_mimic': 8, 'hcc': 2, 'synth_hcc': 2}
RANK_L = 2            # block multilinear rank L (rank-(L,L,1))
RIDGE = 1.0           # relative ALS ridge (validated in covert_gonogo)
CLIP_C = None         # DP clip on the per-party score (None = set per-run from score scale)
DP_SIGMA = 0.0        # DP Gaussian std on the aggregated score (0 = non-private upper bound)
K_FOLDS = 5
SEEDS = 10
SPLIT_RATIO = 0.8     # grouped-by-patient, stratified-by-label (HCCNet GroupStratifiedSplit)


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR
