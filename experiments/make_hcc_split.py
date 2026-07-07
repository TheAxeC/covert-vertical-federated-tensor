"""
Deterministic, auditable PATIENT-LEVEL split for the COVERT HCC pipeline - the reproducibility
HCCNet never saved (its `GroupStratifiedSplit.select_best_split` uses the global RNG and was never
persisted; see RESULTS.md). Splitting at the PATIENT level (not study) means stage-3 finetune (via the
`load_data` patch) and OOF embedding extraction assign every study to test/fold by `patient_id` from
THIS file, so the partition is identical and leak-free by construction.

Writes  ~/projects/covert/splits/hcc_split_h<H>_seed<seed>.json:
    { meta: {...}, test_pids: [...], folds: [ {train_pids:[...], val_pids:[...]}, ... ] }

Leak-free protocol it enables: for CV fold k, finetune a backbone on `folds[k].train_pids`, then embed
`folds[k].val_pids` with THAT backbone (it never saw them) -> OOF embeddings over all dev patients; the
held-out `test_pids` are seen by NO finetune run (embed with any/ensemble dev backbone).

Run:  python -m experiments.make_hcc_split  [--horizon 12 --seed 1234 --k-folds 5 --test-frac 0.25]
"""
import os
import csv
import json
import argparse

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.environ.get('HCCNET_DATA_DIR', '/deepstore/datasets/bms/hcc_study'))
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--test-frac', type=float, default=0.25)
    ap.add_argument('--out-dir', default=os.path.expanduser('~/projects/covert/splits'))
    a = ap.parse_args()

    lab = os.path.join(a.data_dir, 'labels', f'labels_{a.horizon}months.csv')
    rows = list(csv.DictReader(open(lab)))
    # patient-level label = max study label (incident-HCC within horizon for ANY of the patient's studies).
    # patient id is stored STRIPPED ("PT_002" -> "002") to match HCCNet's label_df['patient_id'] format
    # (DatasetPreprocessor strips the "PT_" prefix), so the load_data patch + OOF extractor match by id.
    def _pid(raw):
        return raw.split('_')[-1]
    plab = {}
    for r in rows:
        plab[_pid(r['id'])] = max(plab.get(_pid(r['id']), 0), int(float(r['label'])))
    pids = np.array(sorted(plab))
    y = np.array([plab[p] for p in pids])

    # held-out test (stratified by patient label)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=a.test_frac, random_state=a.seed)
    dev_idx, test_idx = next(sss.split(pids, y))
    test_pids = sorted(pids[test_idx].tolist())
    dev_pids, dev_y = pids[dev_idx], y[dev_idx]

    # k-fold stratified on dev
    skf = StratifiedKFold(n_splits=a.k_folds, shuffle=True, random_state=a.seed)
    folds = []
    for tr, va in skf.split(dev_pids, dev_y):
        folds.append({'train_pids': sorted(dev_pids[tr].tolist()),
                      'val_pids': sorted(dev_pids[va].tolist())})

    meta = dict(horizon=a.horizon, seed=a.seed, k_folds=a.k_folds, test_frac=a.test_frac,
                label_file=lab, n_patients=int(len(pids)), n_pos_patients=int(y.sum()),
                n_test=len(test_pids), test_prev=round(float(y[test_idx].mean()), 3),
                n_dev=int(len(dev_pids)), dev_prev=round(float(dev_y.mean()), 3))
    os.makedirs(a.out_dir, exist_ok=True)
    outp = os.path.join(a.out_dir, f'hcc_split_h{a.horizon}_seed{a.seed}.json')
    json.dump(dict(meta=meta, test_pids=test_pids, folds=folds), open(outp, 'w'), indent=1)

    print('wrote', outp)
    print(json.dumps(meta, indent=1))
    for k, f in enumerate(folds):
        vy = [plab[p] for p in f['val_pids']]
        print(f"  fold{k}: train_pat={len(f['train_pids'])} val_pat={len(f['val_pids'])} "
              f"val_pos={sum(vy)} ({np.mean(vy):.3f})")


if __name__ == '__main__':
    main()
