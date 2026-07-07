# COVERT: vertical federated block-term tensor regression

Code for *"When Masking Meets Sparsity: COVERT for Private, Communication-Efficient Vertical Federated Tensor Regression"* (Faes et al., IEEE J-BHI). Parties hold different modalities of the same patients (imaging / ECG / labs / vitals), with the label held by one active party. COVERT fits one coupled rank-(Lr,Lr,1) block-term regression in which only a low-dimensional, DP-privatized **coupling score** crosses party boundaries, while every party's feature-mode factors stay private. It is the supervised, multilinear, vertical-federated generalization of privacy-preserving PLS: at **parity** accuracy with the centralized model and the linear vertical-PLS baseline, it adds a **differential-privacy guarantee independent of the feature dimension**, **dimension-free communication** (only the cohort-sized score crosses), and **sparse, interpretable multiway components**. The algorithm is in `core/covert.py` (`fit_covert` / `predict_covert`).

## Layout (a Python package; run from this directory)

```
core/        covert.py (the algorithm: rank-(L,L,1) block-term fit + the DP score mechanism, sensitivity 2c),
             baselines.py (P3LS, FBTTR, centralized, concat, single-party), config.py (env-overridable paths),
             align.py (entity-alignment / PSI primitive), data_{hcc,mimic,synth}.py (loaders + a synth fallback)
experiments/ run.py (the driver: grouped-stratified CV, AUROC, the comparison table + rank/party/DP ablations,
             and the paste-ready manuscript macro block), extract_{hcc_embeddings_ft,mimic_features}.py (the
             feature extractors), make_hcc_split.py, sanity_dwi_probe.py
hpc/         run_all.sh (submit the whole from-scratch chain) + the stage1-3 + extract + MIMIC SLURM jobs
results/     generated outputs (not in the deposit)
paper.py     run the synthetic end-to-end pipeline + the HPC feature-extraction map
```

Paths resolve through `core/config.py` (every path env-overridable; `COVERT_HCC_FEATURES` / `COVERT_MIMIC_FEATURES` point at the feature `.npz`, `COVERT_RESULTS_DIR` at the outputs).

## What COVERT is (the forced design)

The privacy precedent (P3LS) is private because PLS reduces to an SVD and orthogonal masks pass through an SVD unchanged. Block-term regression does not: its component selection soft-thresholds a sparse core, and orthogonal masking does not commute with soft-thresholding. The paper's no-go theorem shows the only orthogonal masks that preserve a thresholded support are signed permutations (which hide nothing), so a private sparse vertical decomposition must keep the sparse decomposition LOCAL and move privacy onto the cohort-sized shared score (Route B, the primary design). This preserves the interpretable sparse factors and confines cryptography to an m-dimensional exchange, giving the dimension-free DP and communication.

## Reproduce

COVERT's solver comparison is light and runs anywhere; the real features (HCC backbone embeddings + MIMIC-IV-ECG) are extracted on the cluster from governed data, so they are not in the repo. The HCC imaging features are leak-free out-of-fold embeddings: COVERT re-trains the imaging pipeline in its OWN fork with a controlled, saved split before embedding, and never touches the HCCNet deposit. Start with the synthetic path, then the full reproduction.

**A. Synthetic end-to-end** (no data, no cluster - the quickest check; stand-ins are loudly labelled, never reported as results):
```bash
pip install -r requirements.txt                  # numpy scipy scikit-learn pandas torch monai nibabel scikit-image wfdb
python paper.py                                  # COVERT vs baselines on a MIMIC-shaped stand-in
python -m experiments.run --dataset synth_hcc    # 2-party HCC-shaped stand-in
```

**B. Full reproduction** (regenerate every real number on the UTwente HPC) - **one command**:

```bash
bash hpc/run_all.sh        # watch: squeue -u $USER
```

This submits the whole SLURM DAG and exits in seconds; the DAG then runs unattended. It builds the HCC imaging features from scratch (per modality group: DINO SSL pretrain -> SSL-decoder pretrain -> supervised finetune -> leak-free OOF embedding extract, afterok-chained), the independent MIMIC feature job, and a **final analysis job** (afterok both features) that regenerates EVERY reported number and figure: it runs all six generators (accuracy + ablations, paired-Wilcoxon/TOST significance, component stability, protocol cost, DP accounting) and the figure builder, emitting every manuscript `\renewcommand` into `results/*.txt` and every panel into `manuscript/figures/`. Each `\newcommand` in `main.tex` has a matching emitted `\renewcommand` (verified). The headline rank `R` is resolved per-dataset (HCC `R=2` - it overfits at higher `R` since d>>n; MIMIC `R=8` - the convergence plateau) and every generator runs at `seeds=10`, so there is no `--R`/`--seeds` flag and no by-hand step. Wall-clock and PSI timings are machine-dependent and reported as indicative.

Prerequisites (governed data + a COVERT-owned fork, not redistributed - see Data contract): the imaging pipeline fork at `~/projects/covert/hccnet_pipeline` (rsynced HCCNet source, no weights), the controlled split `~/projects/covert/splits/hcc_split_h12_seed1234.json` (build once with `python -m experiments.make_hcc_split`), governed HCC + MIMIC data under `~/data`, and the `~/envs/hccnet` env (needs `numpy<2`). Nothing pre-trained is assumed - stage1 DINO builds the SSL backbone in-chain.

The pipeline stage by stage:

| stage | job / command | output | manuscript |
|---|---|---|---|
| HCC imaging (per group dwi, t1iop) | `stage1_ssl_covert.slurm` -> `stage2_decoder_covert.slurm` -> `stage3_finetune_covert.slurm` -> `extract_embeddings_covert.slurm` | `hcc_embeddings.npz` (leak-free OOF) | imaging arm |
| MIMIC | `extract_mimic_covert.slurm` | `mimic_features.npz` (ecg 12x8 + labs + vitals) | ECG-labs-vitals arm |
| analysis (all numbers + figures) | `analysis_covert.slurm` -> `run` (accuracy+ablations) + `significance` (Wilcoxon/TOST/CI) + `ablations_extra` (stability) + `bench_protocol` (comms/wall/PSI/Route-A-B) + `dp_curve` (epsilon) + `make_figures` | `results/*.txt` (every `\renewcommand`) + `manuscript/figures/*.pdf` | all tables + figures |

Baselines are P3LS, FBTTR, centralized (upper bound), concat, and best-single-party at matched protocol (grouped-stratified CV, AUROC).

## Data contract

Every loader (`data_{hcc,mimic,synth}.py`) returns a row-aligned cohort:
```python
dict(parties=[X_p, ...], party_names=[...], y=(n,), groups=(n,), name=str)
```
Each party array is `(n, d1, d2)` for a genuinely multiway party (e.g. ECG: patient x lead x time-feature) or `(n, d)` for a tabular party (passed as `(n, d, 1)`, a linear party). Datasets are free / public under credentialed access (MIMIC-IV-ECG and MIMIC-IV via PhysioNet; the HCC cohort is governed by the hosting centre); no raw data is redistributed. The full proofs of the privacy and convergence guarantees are in the companion theory paper.

License: MIT (`LICENSE`).
