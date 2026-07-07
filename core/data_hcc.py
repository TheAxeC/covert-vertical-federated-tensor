"""
HCCNet adapter for COVERT (Twente sign-off obtained).

VERTICAL SPLIT (decided 2026-06-18): no clinical/tabular table exists on disk, so the two parties
are IMAGING SUB-VIEWS over the same studies - an honest multi-view vertical federation:
    dwi   : diffusion party    -- DWI b0/b150/b400/b800            (4 b-values)
    anat  : anatomical party   -- T1W_IP/T1W_OOP + T2W_TEL/T2W_TES (4 volumes)
COVERT is a TENSOR-regression method, so it consumes FEATURES, not raw voxels: each party is a
3-way tensor (studies x volumes x feature-dim) of HCCNet-backbone embeddings per volume.

Sample unit = a labelled imaging study; label = incident-HCC within the chosen horizon
(config.HCC_LABEL_HORIZON, default 12 months); splits are GROUPED BY PATIENT (positives cluster
in few patients, cohort is heavily imbalanced -> report PR-AUC + calibration, not AUROC alone).

EXPECTED FEATURE FILE  (config.HCC_FEATURES, an .npz) with arrays:
    dwi      : (N, 4, F)   float   # DWI b-values embedded; F = embed dim
    anat     : (N, 4, F)   float   # T1W_IP/OOP + T2W_TEL/TES embedded
    y        : (N,)        int     # incident-HCC label at the chosen horizon
    groups   : (N,)        int/str # patient id PT_xxx (grouped split; studies of one patient stay together)

HOW TO PRODUCE IT (run inside ~/projects/hccnet on the HPC, where the data lives):
    1. Load the pretrained DINO backbone (HCCNet stage1); it supports multi-channel modalities.
    2. For each labelled study, embed each volume in config.HCC_SEQUENCES -> a feature vector;
       stack the 4 DWI -> dwi (4,F) and the 4 T1W/T2W -> anat (4,F). Standardize per (volume,feature).
    3. Join the incident-HCC label at config.HCC_LABEL_HORIZON; carry PT_xxx as groups.
    4. np.savez(config.HCC_FEATURES, dwi=..., anat=..., y=..., groups=...)
   (The extractor `extract_hcc_features.py` is the one piece that must run on the HPC; downstream
    runs anywhere.)

If the file is absent this loader emits a clearly-labelled SYNTHETIC stand-in of the right
shape so the pipeline still runs - it must NOT be mistaken for real results.
"""
import os
import warnings
import numpy as np
from core import config
from core.data_synth import make_synth


def load_hcc(seed=0):
    path = config.HCC_FEATURES
    if not os.path.exists(path):
        warnings.warn(f"[data_hcc] {path} not found -> SYNTHETIC HCC-shaped stand-in. "
                      f"NOT real results. Produce features per data_hcc.py header.")
        d = make_synth(name='SYNTH-STANDIN', kind='hcc', n=174, seed=seed)  # ~12mo labelled n
        return d
    z = np.load(path, allow_pickle=True)
    dwi, anat = z['dwi'].astype(float), z['anat'].astype(float)
    y, groups = z['y'].astype(int), z['groups']
    assert dwi.shape[0] == anat.shape[0] == y.shape[0], "HCC feature row mismatch"
    return dict(parties=[dwi, anat], party_names=['dwi', 'anat'],
                y=y, groups=groups, name='hccnet')
