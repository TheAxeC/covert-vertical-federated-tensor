"""
MIMIC-IV(-ECG) feature extractor for COVERT - builds the 3-party vertical-FL feature file
(run on the HPC where the data lives). Produces config.MIMIC_FEATURES with the data_mimic.py contract:

    ecg     : (N, 12, F)  float   # per-lead ECG features (band-power + time stats) - the multiway party
    labs    : (N, L)      float   # per-subject lab aggregates (last value before index ECG) - tabular
    vitals  : (N, V)      float   # OMR vitals + age + sex - tabular
    y        : (N,)       int     # 1-year all-cause mortality (death within HORIZON days of index ECG)
    groups   : (N,)       int     # subject_id (grouped split)
    *_names  : feature names (interpretability)

LABEL = 1-year all-cause mortality (patients.dod within HORIZON of the index ECG). Deliberately NOT
incident-AF - that is the separate dries/afnet task; COVERT uses a generic multi-modal outcome to avoid
overlap. Cohort = one INDEX ECG (earliest) per subject that has ECG + labs + vitals + demographics.

ECG access is a self-contained vendored copy of afnet/core/data.py's lazy zip reader (we do NOT import or
edit the afnet project). Paths default to the cluster ~/data locations; override via env.

Run from the package root ~/projects/covert:
    python -m experiments.extract_mimic_features                 # full (needs hosp files)
    python -m experiments.extract_mimic_features --ecg-smoke --limit 8   # ECG party only, testable now
"""
import os
import sys
import zipfile
import argparse
import warnings

import numpy as np
import pandas as pd

# --- paths (cluster ~/data; env-overridable) -------------------------------------------------
_ZIP_NAME = 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
_ZIP_CANDS = [os.path.expanduser(p) for p in (f'~/data/{_ZIP_NAME}', f'~/data/archive/{_ZIP_NAME}')]
ECG_ZIP = os.environ.get('COVERT_MIMIC_ECG_ZIP',
                         next((p for p in _ZIP_CANDS if os.path.exists(p)), _ZIP_CANDS[0]))
ECG_META = os.environ.get('COVERT_MIMIC_ECG_META', os.path.expanduser('~/data/mimic-ecg'))      # record_list.csv
WAVE_CACHE = os.path.join(ECG_META, 'files')
HOSP = os.environ.get('COVERT_MIMIC_HOSP', os.path.expanduser('~/data/mimic-iv/hosp'))            # *.csv.gz (chain wrote here)
OUT = os.environ.get('COVERT_MIMIC_FEATURES', os.path.expanduser('~/data/mimic-iv/covert/mimic_features.npz'))
ZIP_ROOT = 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
FS = 500
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
HORIZON_DAYS = int(os.environ.get('COVERT_MIMIC_HORIZON', '365'))
ECG_FEAT_NAMES = ['bp_0.5-5Hz', 'bp_5-15Hz', 'bp_15-40Hz', 'bp_40-100Hz', 'std', 'rms', 'range', 'iqr']
# common labs (itemid -> short name); last value before index ECG
LAB_ITEMS = {50862: 'albumin', 50912: 'creatinine', 50971: 'potassium', 50983: 'sodium',
             51006: 'urea_n', 50902: 'chloride', 50882: 'bicarb', 50931: 'glucose',
             51221: 'hematocrit', 51222: 'hemoglobin', 51265: 'platelet', 51301: 'wbc',
             50960: 'magnesium', 50893: 'calcium', 51237: 'inr', 50820: 'ph'}
VITAL_NAMES = ['sbp', 'dbp', 'bmi', 'weight_kg', 'height_in']

_zf = None


def _zip():
    global _zf
    if _zf is None:
        _zf = zipfile.ZipFile(ECG_ZIP, 'r')
    return _zf


def load_ecg(rel_path):
    """Vendored from afnet/core/data.py: extract .hea/.dat from the zip (cache locally), wfdb read -> (T,12)."""
    import wfdb
    base = os.path.join(os.path.dirname(WAVE_CACHE), rel_path)   # ~/data/mimic-ecg/files/...
    hea, dat = base + '.hea', base + '.dat'
    if not (os.path.exists(hea) and os.path.exists(dat)):
        os.makedirs(os.path.dirname(base), exist_ok=True)
        zf = _zip()
        for ext, dst in (('.hea', hea), ('.dat', dat)):
            with zf.open(f'{ZIP_ROOT}/{rel_path}{ext}') as src, open(dst, 'wb') as out:
                out.write(src.read())
    rec = wfdb.rdrecord(base)
    return np.nan_to_num(np.asarray(rec.p_signal, dtype=float), nan=0.0), int(rec.fs)   # (T,12)


def _lead_features(x, fs):
    """8 interpretable features for one lead (T,): 4 log band-powers + std/rms/range/iqr."""
    from scipy.signal import welch
    f, p = welch(x, fs=fs, nperseg=min(len(x), 1024))
    bands = [(0.5, 5), (5, 15), (15, 40), (40, 100)]
    bp = [np.log1p(np.trapz(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])) for lo, hi in bands]
    return np.array(bp + [x.std(), np.sqrt((x ** 2).mean()), float(np.ptp(x)),
                          np.subtract(*np.percentile(x, [75, 25]))], dtype=np.float32)


def ecg_features(rel_path):
    """(12, 8) features for one record, or None on failure."""
    try:
        sig, fs = load_ecg(rel_path)            # (T, 12)
    except Exception as e:                      # noqa: BLE001
        warnings.warn(f'ecg load fail {rel_path}: {e}')
        return None
    if sig.shape[1] != 12:
        return None
    return np.stack([_lead_features(sig[:, c], fs) for c in range(12)], 0)   # (12, 8)


def index_ecgs(limit=0, inpatient=True):
    """One INDEX ECG per subject -> df[subject_id, ecg_time, path]. inpatient=True picks the first ECG that
    falls INSIDE a hospital admission window (admissions.csv.gz), so labs/vitals are contemporaneous (the
    outpatient earliest-ECG default left labs 57% missing); otherwise the earliest ECG."""
    rl = pd.read_csv(os.path.join(ECG_META, 'record_list.csv'), usecols=['subject_id', 'ecg_time', 'path'])
    rl['ecg_time'] = pd.to_datetime(rl['ecg_time'], errors='coerce')
    rl = rl.dropna(subset=['ecg_time']).sort_values('ecg_time')
    if inpatient:
        adm = pd.read_csv(os.path.join(HOSP, 'admissions.csv.gz'),
                          usecols=['subject_id', 'admittime', 'dischtime'],
                          parse_dates=['admittime', 'dischtime'])
        m = rl.merge(adm, on='subject_id', how='inner')
        m = m[(m['ecg_time'] >= m['admittime']) & (m['ecg_time'] <= m['dischtime'])]
        idx = m.sort_values('ecg_time').groupby('subject_id', as_index=False).first()
    else:
        idx = rl.groupby('subject_id', as_index=False).first()
    return idx.head(limit) if limit else idx


# --- tabular parties (need the hosp files) ---------------------------------------------------
def demographics_and_label(subjects, idx_time):
    """patients.csv.gz -> age, sex, and y = death within HORIZON days of index ECG."""
    pt = pd.read_csv(os.path.join(HOSP, 'patients.csv.gz'),
                     usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod'])
    pt = pt[pt['subject_id'].isin(subjects)].copy()
    pt['dod'] = pd.to_datetime(pt['dod'], errors='coerce')
    pt['sex'] = (pt['gender'] == 'M').astype(int)
    pt = pt.merge(idx_time.rename('ecg_time'), left_on='subject_id', right_index=True, how='left')
    dt = (pt['dod'] - pt['ecg_time']).dt.days
    pt['y'] = ((dt >= 0) & (dt <= HORIZON_DAYS)).astype(int)
    return pt.set_index('subject_id')[['anchor_age', 'sex', 'y']]


def lab_features(subjects, idx_time, window_days=7):
    """labevents.csv.gz (chunked) -> for each LAB_ITEMS analyte, the value NEAREST the index ECG within
    +/- window_days -> (subjects x L). Wider two-sided window (was strictly-before) captures the admission's
    labs and cuts the ~57% missingness."""
    cols = ['subject_id', 'itemid', 'charttime', 'valuenum']
    keep_items = set(LAB_ITEMS)
    rows = {s: {} for s in subjects}                    # subject -> {itemid: (abs_days, value)}
    subj_set = set(subjects)
    for ch in pd.read_csv(os.path.join(HOSP, 'labevents.csv.gz'), usecols=cols,
                          parse_dates=['charttime'], chunksize=2_000_000):
        ch = ch[ch['itemid'].isin(keep_items) & ch['subject_id'].isin(subj_set)].dropna(subset=['valuenum'])
        for s, it, t, v in ch[['subject_id', 'itemid', 'charttime', 'valuenum']].itertuples(index=False):
            dt = abs((t - idx_time.get(s, t)).days)
            if dt <= window_days:
                prev = rows[s].get(it)
                if prev is None or dt < prev[0]:        # keep the value nearest the ECG
                    rows[s][it] = (dt, v)
    order = list(LAB_ITEMS)
    mat = np.array([[rows[s].get(it, (None, np.nan))[1] for it in order] for s in subjects], dtype=float)
    return mat, [LAB_ITEMS[i] for i in order]


def vital_features(subjects, idx_time):
    """omr.csv.gz -> per-subject SBP/DBP/BMI/weight/height nearest the index ECG -> (subjects x 5)."""
    omr = pd.read_csv(os.path.join(HOSP, 'omr.csv.gz'),
                      usecols=['subject_id', 'chartdate', 'result_name', 'result_value'])
    omr = omr[omr['subject_id'].isin(set(subjects))].copy()
    omr['chartdate'] = pd.to_datetime(omr['chartdate'], errors='coerce')
    def _num(v):
        try:
            return float(str(v).split('/')[0])
        except Exception:
            return np.nan
    out = {s: [np.nan] * 5 for s in subjects}
    best = {s: [np.inf] * 5 for s in subjects}          # abs-days to the ECG, per slot -> keep nearest
    name_map = {'Blood Pressure': (0, 1), 'BMI (kg/m2)': (2,), 'Weight (Lbs)': (3,), 'Height (Inches)': (4,)}
    for s, g in omr.groupby('subject_id'):
        et = idx_time.get(s)
        for _, r in g.iterrows():
            rn = str(r['result_name'])
            dt = abs((r['chartdate'] - et).days) if et is not None and pd.notna(r['chartdate']) else 0
            for key, slots in name_map.items():
                if rn.startswith(key.split(' (')[0]):
                    parts = str(r['result_value']).split('/')
                    for j, slot in enumerate(slots):
                        if dt < best[s][slot]:          # OMR is stable -> nearest chartdate to the ECG (any date)
                            try:
                                out[s][slot] = float(parts[j]) if key == 'Blood Pressure' else _num(r['result_value'])
                                best[s][slot] = dt
                            except Exception:
                                pass
    mat = np.array([out[s] for s in subjects], dtype=float)
    mat[:, 3] = mat[:, 3] * 0.4536          # lbs -> kg
    return mat, VITAL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ecg-smoke', action='store_true', help='ECG party only (no hosp needed) - testable now')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--window-days', type=int, default=7, help='lab window (+/- days) around the index ECG')
    ap.add_argument('--outpatient', action='store_true', help='use earliest ECG (default = first INPATIENT ECG)')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    idx = index_ecgs(limit=a.limit, inpatient=not a.outpatient)
    print(f'[mimic] index ECGs: {len(idx)} subjects (inpatient={not a.outpatient}, lab-window=+/-{a.window_days}d)', flush=True)

    # ECG party (the heavy, multiway one) - parallel over records
    from concurrent.futures import ProcessPoolExecutor
    paths = idx['path'].tolist()
    feats = list(ProcessPoolExecutor(max_workers=a.workers).map(ecg_features, paths, chunksize=8))
    ok = [i for i, f in enumerate(feats) if f is not None]
    idx = idx.iloc[ok].reset_index(drop=True)
    ecg = np.stack([feats[i] for i in ok], 0).astype(np.float32)        # (N,12,8)
    subjects = idx['subject_id'].to_numpy()
    idx_time = pd.Series(idx['ecg_time'].values, index=subjects)
    print(f'[mimic] ECG features: {ecg.shape} (kept {len(ok)}/{len(paths)})', flush=True)

    if a.ecg_smoke:
        sout = a.out.replace('.npz', '_ecgsmoke.npz')
        os.makedirs(os.path.dirname(sout) or '.', exist_ok=True)
        np.savez(sout, ecg=ecg, groups=subjects, ecg_names=np.array(ECG_FEAT_NAMES))
        print(f'[mimic] ECG-SMOKE wrote {sout}  ecg={ecg.shape}', flush=True)
        return

    dem = demographics_and_label(subjects, idx_time)
    labs, lab_names = lab_features(subjects, idx_time, window_days=a.window_days)
    vit, vit_names = vital_features(subjects, idx_time)
    print(f'[mimic] sparsity: labs NaN {np.isnan(labs).mean():.1%}, vitals NaN {np.isnan(vit).mean():.1%}', flush=True)
    y = dem.reindex(subjects)['y'].fillna(0).to_numpy().astype(int)
    vitals = np.column_stack([vit, dem.reindex(subjects)['anchor_age'].to_numpy(),
                              dem.reindex(subjects)['sex'].to_numpy()])
    vitals_names = vit_names + ['age', 'sex']
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out, ecg=ecg, labs=labs.astype(np.float32), vitals=vitals.astype(np.float32),
             y=y, groups=subjects, ecg_names=np.array(ECG_FEAT_NAMES),
             lab_names=np.array(lab_names), vitals_names=np.array(vitals_names))
    print(f'[mimic] wrote {a.out}  N={len(y)} pos={int(y.sum())} ({y.mean():.3f})  '
          f'ecg={ecg.shape} labs={labs.shape} vitals={vitals.shape}', flush=True)

    # --- source-cohort descriptor macros, regenerated from the raw record list (not the analysed subset) ---
    rl_all = pd.read_csv(os.path.join(ECG_META, 'record_list.csv'), usecols=['subject_id'])
    n_ecg, n_subj = len(rl_all), int(rl_all['subject_id'].nunique())
    try:
        _, fs_seen = load_ecg(idx['path'].iloc[0]); fs_seen = int(fs_seen)
    except Exception:
        fs_seen = 500
    def _thou(x):
        return f"{x:,}".replace(',', '{,}')
    print("\n% source-cohort macros (MIMIC-IV-ECG):")
    print(f"\\renewcommand{{\\mimicEcgs}}{{{_thou(n_ecg)}}}  \\renewcommand{{\\mimicPatients}}{{{_thou(n_subj)}}}")
    print(f"\\renewcommand{{\\mimicFs}}{{{fs_seen}}}")


if __name__ == '__main__':
    main()
