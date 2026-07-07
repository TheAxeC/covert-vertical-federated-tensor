"""
HCC multi-sequence feature extractor for COVERT  (run on the HPC where the data lives).

Design (decided 2026-06-18):
  * Task = INCIDENT HCC within a horizon (config.HCC_LABEL_HORIZON, default 12 mo). Because the
    outcome is FUTURE onset, at baseline many positives have no lesion yet -> the predictive
    substrate is whole-liver / parenchymal texture (cirrhosis), NOT a lesion mask. And SEG lives
    only in T1W space, so masking other sequences would need cross-sequence registration. We
    therefore use REGISTRATION-FREE, foreground-masked radiomics computed per volume in its OWN
    native grid -> modality-agnostic, comparable across DWI/T1W/T2W, no registration assumption.
  * Two vertical parties (imaging sub-views), config.HCC_PARTIES:
        dwi  : DWI_b0/b150/b400/b800            -> tensor (N, 4, F)
        anat : T1W_IP/T1W_OOP + T2W_TEL/T2W_TES -> tensor (N, 4, F)
  * Features per volume (F, all scanner-robust / intensity-normalized, all NAMED for the
    interpretability claim): first-order distribution shape on the within-mask z-normalized
    intensities, GLCM texture, and gradient/edge energy. See FEATURE_NAMES.

Sample unit = a labelled imaging study with ALL 8 volumes present; groups = patient id (PT_xxx).
Output: np.savez(config.HCC_FEATURES, dwi, anat, y, groups, feat_names, vols_dwi, vols_anat).

Run (cluster, hccnet conda env):
    module load anaconda3/2024.02 && source activate ~/envs/hccnet
    COVERT_HCC_FEATURES=~/projects/covert/hcc_features.npz python extract_hcc_features.py [--workers 8]
"""
import os, csv, argparse, warnings
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.feature import graycomatrix, graycoprops
from core import config

VOLS_DWI  = config.HCC_PARTIES['dwi']
VOLS_ANAT = config.HCC_PARTIES['anat']
ALL_VOLS  = VOLS_DWI + VOLS_ANAT

FEATURE_NAMES = [
    'fo_skew', 'fo_kurt', 'fo_p10', 'fo_p25', 'fo_p50', 'fo_p75', 'fo_p90',
    'fo_iqr', 'fo_mad', 'fo_entropy', 'fo_energy',
    'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy',
    'glcm_correlation', 'glcm_ASM',
    'grad_mean', 'grad_std', 'lap_energy',
    'fg_fraction', 'vol_logsize',
]
F = len(FEATURE_NAMES)


def _foreground(vol):
    pos = vol[vol > 0]
    if pos.size < 50:
        return vol > vol.mean()
    try:
        t = threshold_otsu(pos)
    except Exception:
        t = np.percentile(pos, 25)
    m = vol > t
    if m.sum() < 50:
        m = vol > np.percentile(pos, 10)
    return m


def _glcm_feats(znorm, mask):
    """GLCM texture averaged over a few central axial slices (32-level, 4 angles)."""
    levels = 32
    q = np.clip((znorm + 3.0) / 6.0, 0, 1)          # map z in [-3,3] -> [0,1]
    q = (q * (levels - 1)).astype(np.uint8)
    zc = np.where(mask.any(axis=(0, 1)))[0]
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    if zc.size == 0:
        return [0.0] * len(props)
    mid = int(zc[len(zc) // 2])
    sel = [s for s in range(mid - 3, mid + 4) if 0 <= s < q.shape[2] and mask[:, :, s].sum() > 30]
    if not sel:
        sel = [mid]
    acc = {p: [] for p in props}
    for s in sel:
        sl = q[:, :, s].copy()
        sl[~mask[:, :, s]] = 0
        g = graycomatrix(sl, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=levels, symmetric=True, normed=True)
        for p in props:
            acc[p].append(float(graycoprops(g, p).mean()))
    return [float(np.mean(acc[p])) for p in props]


def _features_one(vol):
    vol = np.asarray(vol, dtype=np.float32)
    mask = _foreground(vol)
    x = vol[mask]
    if x.size < 50:
        return np.zeros(F, dtype=np.float32)
    mu, sd = x.mean(), x.std() + 1e-6
    z = (x - mu) / sd
    p10, p25, p50, p75, p90 = np.percentile(z, [10, 25, 50, 75, 90])
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean() - 3.0)
    hist, _ = np.histogram(z, bins=32, range=(-3, 3), density=True)
    hist = hist / (hist.sum() + 1e-9)
    entropy = float(-(hist * np.log(hist + 1e-9)).sum())
    energy = float((z ** 2).mean())
    iqr = float(p75 - p25)
    mad = float(np.abs(z - np.median(z)).mean())
    znorm_vol = (vol - mu) / sd
    glcm = _glcm_feats(znorm_vol, mask)
    gm = ndimage.gaussian_gradient_magnitude(znorm_vol, sigma=1.0)
    grad_mean, grad_std = float(gm[mask].mean()), float(gm[mask].std())
    lap = ndimage.laplace(znorm_vol)
    lap_energy = float((lap[mask] ** 2).mean())
    fg_fraction = float(mask.mean())
    vol_logsize = float(np.log1p(mask.sum()))
    return np.array([skew, kurt, p10, p25, p50, p75, p90, iqr, mad, entropy, energy,
                     *glcm, grad_mean, grad_std, lap_energy, fg_fraction, vol_logsize],
                    dtype=np.float32)


def _study_features(study_dir):
    """Return (dwi (4,F), anat (4,F)) for one study, or None on any load failure."""
    out = {}
    for v in ALL_VOLS:
        p = os.path.join(study_dir, v + '.nii.gz')
        try:
            vol = nib.load(p).get_fdata(dtype=np.float32)
        except Exception as e:
            warnings.warn(f"load fail {p}: {e}")
            return None
        out[v] = _features_one(vol)
    dwi = np.stack([out[v] for v in VOLS_DWI], 0)
    anat = np.stack([out[v] for v in VOLS_ANAT], 0)
    return dwi, anat


def _worker(args):
    pid, study_dir, label = args
    try:
        fe = _study_features(study_dir)
    except Exception as e:
        warnings.warn(f"study fail {study_dir}: {e}")
        return None
    if fe is None:
        return None
    return pid, fe[0], fe[1], int(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N studies')
    args = ap.parse_args()

    root = config.HCC_DATA_DIR
    horizon = config.HCC_LABEL_HORIZON
    nif = os.path.join(root, 'nifti')
    lab = os.path.join(root, 'labels', f'labels_{horizon}months.csv')
    rows = list(csv.DictReader(open(lab)))

    jobs = []
    for r in rows:
        d = os.path.join(nif, f"{r['id']}_{int(float(r['observation'])):03d}")
        if not os.path.isdir(d):
            continue
        if not all(os.path.exists(os.path.join(d, v + '.nii.gz')) for v in ALL_VOLS):
            continue
        jobs.append((r['id'], d, r['label']))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[extract] horizon={horizon}mo  studies with all {len(ALL_VOLS)} vols = {len(jobs)}  "
          f"(workers={args.workers})", flush=True)

    results = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(_worker, jobs, chunksize=2)):
            if res is not None:
                results.append(res)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(jobs)} done", flush=True)

    groups = np.array([r[0] for r in results])
    dwi = np.stack([r[1] for r in results], 0).astype(np.float32)    # (N,4,F)
    anat = np.stack([r[2] for r in results], 0).astype(np.float32)   # (N,4,F)
    y = np.array([r[3] for r in results], dtype=int)

    # standardize per (party, volume-index, feature) across studies; guard zero-variance
    def _std(a):
        mu = a.mean(0, keepdims=True); sd = a.std(0, keepdims=True)
        return ((a - mu) / np.where(sd < 1e-8, 1.0, sd)).astype(np.float32)
    dwi, anat = _std(dwi), _std(anat)

    out = os.environ.get('COVERT_HCC_FEATURES', config.HCC_FEATURES)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, dwi=dwi, anat=anat, y=y, groups=groups,
             feat_names=np.array(FEATURE_NAMES), vols_dwi=np.array(VOLS_DWI),
             vols_anat=np.array(VOLS_ANAT))
    print(f"[extract] wrote {out}  N={len(y)}  pos={int(y.sum())} ({y.mean():.3f})  "
          f"patients={len(set(groups.tolist()))}  dwi={dwi.shape}  anat={anat.shape}", flush=True)


if __name__ == '__main__':
    main()
