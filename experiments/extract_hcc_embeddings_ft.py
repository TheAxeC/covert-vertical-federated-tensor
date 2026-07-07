"""
OOF leak-free FINE-TUNED-backbone embedding extractor for COVERT (run on the HPC; uses the
COVERT-owned fork, never the Paper-3 deposit).

For each CV fold k of the CONTROLLED split (make_hcc_split.py), a MedNet was finetuned on fold-k's
TRAIN patients (stage3). Here we embed each study with the backbone that did NOT see it:
  * dev study (patient in fold-k val) -> fold-k fine-tuned backbone  (out-of-fold => leak-free)
  * held-out test study               -> mean over all K fold backbones (none trained on test)
Embedding = `MedNet.backbone.forward_features` on the 4-channel supervised test transform -> 384-d
per study per party (a tabular party (N,384); covert.py handles tabular parties as linear).

Output npz (consumed by core/data_hcc.py):  dwi (N,384), anat (N,384), y (N,), groups (N,),
  split (N,)  # fold index 0..K-1 for dev (OOF), -1 for held-out test
Raw embeddings (no global standardization) - downstream standardizes within CV folds.

Run from the package root ~/projects/covert:
    python -m experiments.extract_hcc_embeddings_ft --device cpu
"""
import os
import sys
import csv
import json
import argparse
import warnings

import numpy as np
import torch

from core import config

HCCNET_ROOT = os.environ.get('HCCNET_CODE', os.path.expanduser('~/projects/covert/hccnet_pipeline'))
if HCCNET_ROOT not in sys.path:
    sys.path.append(HCCNET_ROOT)

from models.convnext3d import convnext3d_femto          # noqa: E402
import utils.transforms as T                            # noqa: E402  (T.transforms = supervised transform fn)

VOLS = {'dwi': config.HCC_PARTIES['dwi'], 'anat': config.HCC_PARTIES['anat']}
ALL_VOLS = VOLS['dwi'] + VOLS['anat']
# COVERT party name -> HCCNet training group/tag (the anat party is trained as the 't1iop' group)
PARTY_TAG = {'dwi': 'dwi', 'anat': 't1iop'}
EMB_DIM = 384
WEIGHTS_DIR = os.path.expanduser('~/projects/covert/results/model_weights')


def _pid(raw):
    return raw.split('_')[-1]          # "PT_002" -> "002" (matches the split file + label_df)


def load_ft_backbone(ckpt_path, device):
    """Load the `backbone.*` of a fine-tuned MedNet checkpoint into a 4-channel femto ConvNeXt3d."""
    sd = torch.load(ckpt_path, map_location='cpu')
    bb = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    if not bb:
        raise RuntimeError(f"{ckpt_path}: no backbone.* keys")
    for use_v2 in (False, True):
        m = convnext3d_femto(in_chans=4, kernel_size=3, use_v2=use_v2, eps=1e-5)
        m.head = torch.nn.Identity()
        missing, unexpected = m.load_state_dict(bb, strict=False)
        if not unexpected and not missing:
            return m.eval().to(device)
    raise RuntimeError(f"{ckpt_path}: could not load backbone cleanly (missing={list(missing)[:3]}, "
                       f"unexpected={list(unexpected)[:3]})")


@torch.no_grad()
def embed(study_dir, group, transform, backbones, device):
    """Embed one study's group volumes with each backbone in `backbones`; return mean over them (384,)
    (one backbone for OOF dev, K for ensemble test), or None on load failure."""
    data = {m: os.path.join(study_dir, m + '.nii.gz') for m in VOLS[group]}
    try:
        img = transform(data)['image']                  # (4,72,72,72)
    except Exception as e:                               # noqa: BLE001
        warnings.warn(f"transform fail {study_dir} [{group}]: {e}")
        return None
    x = torch.as_tensor(np.asarray(img), dtype=torch.float)[None].to(device)   # (1,4,72,72,72)
    feats = [bb.forward_features(x).squeeze(0).float().cpu().numpy() for bb in backbones]
    return np.mean(feats, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default=os.path.expanduser('~/projects/covert/splits/hcc_split_h12_seed1234.json'))
    ap.add_argument('--out', default=os.environ.get('COVERT_HCC_FEATURES',
                                                    os.path.expanduser('~/projects/covert/hcc_embeddings.npz')))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)

    split = json.load(open(args.split))
    K = len(split['folds'])
    pid2fold = {p: -1 for p in split['test_pids']}       # -1 = held-out test
    for k, f in enumerate(split['folds']):
        for p in f['val_pids']:
            pid2fold[p] = k

    # fine-tuned backbones per (party, cv-fold), ENSEMBLING all training seeds of that fold.
    # stage3 saves global fold index = seed*K + k, so cv-fold k's seed-replicas are k, k+K, k+2K, ...
    def _fold_ckpts(tag, k):
        paths, s = [], 0
        while True:
            p = os.path.join(WEIGHTS_DIR, f'weights_fold{k + s * K}_{tag}_femto_paper_ft.pth')
            if os.path.exists(p):
                paths.append(p); s += 1
            else:
                return paths
    backbones = {g: {k: [load_ft_backbone(p, device) for p in _fold_ckpts(PARTY_TAG[g], k)]
                     for k in range(K)} for g in ('dwi', 'anat')}
    tf = {g: T.transforms(dataset='test', modalities=VOLS[g], device='cpu') for g in ('dwi', 'anat')}
    n_seeds = {g: len(backbones[g][0]) for g in ('dwi', 'anat')}
    print(f"[ft-embed] loaded {K} cv-folds x 2 parties, seeds/fold={n_seeds}; device={device}", flush=True)

    # enumerate studies with ALL 8 volumes (rows aligned across the two parties)
    root, horizon = config.HCC_DATA_DIR, config.HCC_LABEL_HORIZON
    nif = os.path.join(root, 'nifti')
    rows = list(csv.DictReader(open(os.path.join(root, 'labels', f'labels_{horizon}months.csv'))))
    jobs = []
    for r in rows:
        d = os.path.join(nif, f"{r['id']}_{int(float(r['observation'])):03d}")
        if os.path.isdir(d) and all(os.path.exists(os.path.join(d, v + '.nii.gz')) for v in ALL_VOLS):
            jobs.append((_pid(r['id']), d, int(r['label'])))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[ft-embed] {len(jobs)} studies with all {len(ALL_VOLS)} vols", flush=True)

    dwi, anat, y, groups, splits = [], [], [], [], []
    for i, (pid, d, label) in enumerate(jobs):
        fold = pid2fold.get(pid)
        if fold is None:
            warnings.warn(f"{pid}: not in split, skipping")
            continue
        bk = backbones  # dev: ensemble the seed-replicas of the study's OOF fold; test: ensemble all
        sel = ((lambda g: bk[g][fold]) if fold >= 0
               else (lambda g: [b for k in range(K) for b in bk[g][k]]))
        e_dwi = embed(d, 'dwi', tf['dwi'], sel('dwi'), device)
        e_anat = embed(d, 'anat', tf['anat'], sel('anat'), device)
        if e_dwi is None or e_anat is None:
            continue
        dwi.append(e_dwi); anat.append(e_anat); y.append(label); groups.append(pid); splits.append(fold)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(jobs)} done, kept {len(y)}", flush=True)

    dwi = np.stack(dwi); anat = np.stack(anat)
    y = np.array(y, dtype=int); groups = np.array(groups); splits = np.array(splits, dtype=int)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, dwi=dwi, anat=anat, y=y, groups=groups, split=splits)
    n_test = int((splits < 0).sum())
    print(f"[ft-embed] wrote {args.out}  N={len(y)}  pos={int(y.sum())} ({y.mean():.3f})  "
          f"dev(OOF)={len(y) - n_test} test={n_test}  dwi={dwi.shape} anat={anat.shape}", flush=True)


if __name__ == '__main__':
    main()
