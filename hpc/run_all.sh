#!/bin/bash
# One command to reproduce COVERT end to end from scratch on the UTwente HPC - every reported number.
# Submits the full HCC imaging chain (per modality group: DINO SSL pretrain -> SSL-decoder pretrain ->
# supervised finetune -> one leak-free OOF embedding extract, afterok-chained), the independent MIMIC
# feature job, and a FINAL comparison job (afterok both features) that emits both datasets' tables, the
# paste-ready macro block, and the seed-level bootstrap margin CIs into results/. Then EXITS in seconds;
# the SLURM DAG runs unattended. No by-hand step, no --R flag (R is resolved per-dataset in run.py).
#
#   bash hpc/run_all.sh
#
# PREREQUISITES (governed data + a COVERT-owned fork; NOT redistributed - see README "Reproduce"):
#   - the imaging pipeline fork  ~/projects/covert/hccnet_pipeline  (rsynced HCCNet source, no weights)
#   - the controlled split  ~/projects/covert/splits/hcc_split_h12_seed1234.json  (build once:
#     `python -m experiments.make_hcc_split`); governed HCC imaging data + MIMIC-IV(-ECG) under ~/data
#   - conda env  ~/envs/hccnet  (needs numpy<2)
# stage1 DINO produces the SSL backbone in-chain, so this is a genuine from-scratch reproduction of the
# imaging arm (nothing pre-trained is assumed); the four stages match the manuscript's job provenance.
set -e
cd "$(dirname "$0")"                     # -> code/hpc (each job cd's to its own absolute workdir internally)
mkdir -p logs "$HOME/projects/covert/logs"   # the stage jobs write --output=logs/... RELATIVE to here (hpc/); it must exist or SLURM fails the job at launch

# --- HCC imaging arm: two modality groups, each DINO SSL -> SSL-decoder -> supervised finetune, then one OOF extract ---
P3IDS=""
for G in dwi t1iop; do
  P1=$(sbatch --parsable stage1_ssl_covert.slurm "$G")
  P2=$(sbatch --parsable --dependency=afterok:$P1 stage2_decoder_covert.slurm "$G")
  P3=$(sbatch --parsable --dependency=afterok:$P2 stage3_finetune_covert.slurm "$G")
  printf 'HCC %-6s dino SSL = %s  ->  decoder = %s  ->  finetune = %s\n' "$G" "$P1" "$P2" "$P3"
  P3IDS="${P3IDS:+$P3IDS:}$P3"
done
EMB=$(sbatch --parsable --dependency=afterok:$P3IDS extract_embeddings_covert.slurm)
echo "HCC        OOF embeddings = $EMB (afterok both finetunes)  ->  ~/projects/covert/hcc_embeddings.npz"

# --- MIMIC arm: independent CPU feature extraction ---
MIM=$(sbatch --parsable extract_mimic_covert.slurm)
echo "MIMIC      features       = $MIM  ->  ~/data/mimic-iv/covert/mimic_features.npz"

# --- final analysis: EVERY reported number + figure, straight into the DAG (afterok BOTH feature jobs) ---
# run.py (accuracy+ablations) + significance (Wilcoxon/TOST/CI) + ablations_extra (component stability) +
# bench_protocol (comms/wall/PSI/Route-A-B/convergence) + dp_curve (epsilon) + make_figures. One command
# reproduces the whole paper; per-dataset R and seeds=10 are baked in (no flags). Every macro is emitted.
ANA=$(sbatch --parsable --dependency=afterok:$EMB:$MIM analysis_covert.slurm)
echo "ANALYSIS   = $ANA (afterok both features)  ->  results/*.txt (all macros) + manuscript/figures/*.pdf"

echo
echo "submitted. watch: squeue -u $USER ; logs: ~/projects/covert/logs/"
echo "when the DAG finishes, every reported number is a \\renewcommand in results/*.txt and every figure in figures/."
