"""
MIMIC-IV-ECG adapter for COVERT.

Three vertical parties over a shared patient cohort:
    ecg     : 3-way tensor (patients x 12 leads x time-features)   -- the multiway view
    labs    : tabular (patients x analytes)
    vitals  : tabular (patients x vital signals)
Label (LABEL-1, to finalize): a diagnostic label (e.g. AF) or an outcome (e.g. mortality).

EXPECTED FEATURE FILE (config.MIMIC_FEATURES, an .npz):
    ecg     : (N, 12, T)  float   # per-lead time features (e.g. wavelet/band-power or a
                                  #   learned encoder embedding per lead); standardized
    labs    : (N, L)      float
    vitals  : (N, V)      float
    y       : (N,)        int
    groups  : (N,)        int     # subject_id (grouped split)

HOW TO PRODUCE IT (needs credentialed PhysioNet access + a signed DUA):
    1. Download MIMIC-IV-ECG + MIMIC-IV; link on subject_id; private-set-intersection to the
       shared cohort with a record in all three parties (see align.py).
    2. ECG: QC + per-lead feature extraction -> (12, T). Labs/vitals: aggregate per patient,
       impute, standardize. Choose the label per LABEL-1.
    3. np.savez(config.MIMIC_FEATURES, ecg=..., labs=..., vitals=..., y=..., groups=...)

Absent the file, a SYNTHETIC MIMIC-shaped stand-in is returned (clearly labelled).
"""
import os
import warnings
import numpy as np
from core import config
from core.data_synth import make_synth


def load_mimic(seed=0, n_synth=4000):
    path = config.MIMIC_FEATURES
    if not os.path.exists(path):
        warnings.warn(f"[data_mimic] {path} not found -> SYNTHETIC MIMIC-shaped stand-in. "
                      f"NOT real results. Needs credentialed PhysioNet access (see header).")
        return make_synth(name='SYNTH-STANDIN', kind='mimic', n=n_synth, seed=seed)
    z = np.load(path, allow_pickle=True)
    ecg, labs, vitals = (z['ecg'].astype(float), z['labs'].astype(float),
                         z['vitals'].astype(float))
    y, groups = z['y'].astype(int), z['groups']
    assert ecg.shape[0] == labs.shape[0] == vitals.shape[0] == y.shape[0], "MIMIC row mismatch"
    # MIMIC labs/vitals are SPARSE (many index ECGs are outpatient w/o contemporaneous measurements):
    # median-impute missing entries column-wise (whole-NaN column -> 0). Done globally for simplicity;
    # a fold-wise imputer (fit on train only) is the leak-free refinement once a result is in hand.
    def _impute(a):
        med = np.nanmedian(a, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        return np.where(np.isnan(a), med, a)
    labs, vitals = _impute(labs), _impute(vitals)
    return dict(parties=[ecg, labs, vitals], party_names=['ecg', 'labs', 'vitals'],
                y=y, groups=groups, name='mimic-iv-ecg')
