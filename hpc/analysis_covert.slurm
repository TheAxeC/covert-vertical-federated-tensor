#!/bin/bash
#SBATCH --job-name=cov_analysis
#SBATCH --partition=main-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# FULL COVERT analysis stage: EVERY reported number + every figure, from the two feature .npz. Runs all
# six generators and teees each to results/, then builds the figure PDFs. This is the last stage of
# run_all.sh, so one command reproduces the whole paper (not just the accuracy table):
#   [1,2] run.py            - accuracy table + rank/party/LOO/DP-sweep ablations (HCC, MIMIC) + margin CIs
#   [3,4] significance.py   - paired Wilcoxon (+Holm) + seed-level TOST equivalence + 95% CI (HCC, MIMIC)
#   [5]   ablations_extra   - component stability vs label-permutation null (compStab/compStabNull)
#   [6]   bench_protocol    - comms (Route A/B), PSI time, wall-clock, Route-A/B accuracy, convergence R
#   [7]   dp_curve          - accounted patient-level epsilon (the DP cost)
#   [8]   make_figures      - component / convergence / scalability / dp_utility .pdf
# Single-threaded BLAS: the solvers make ~1e6 tiny SVD/lstsq calls; multi-threaded BLAS thrashes on them.
# Submit from ~/projects/covert:  sbatch hpc/analysis_covert.slurm    Canonical: covert-applied/code/hpc/.
set -x
source /etc/profile.d/modules.sh 2>/dev/null
module load anaconda3/2024.02
source activate $HOME/envs/hccnet
cd $HOME/projects/covert
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export COVERT_HCC_FEATURES=$HOME/projects/covert/hcc_embeddings.npz
export COVERT_MIMIC_FEATURES=$HOME/data/mimic-iv/covert/mimic_features.npz
mkdir -p results "$HOME/projects/manuscript/figures"
echo "##### COVERT full analysis #####"; date

echo "=== [1/8] accuracy + ablations (HCC) ==="
python -m experiments.run           --dataset hcc   --ablations 2>&1 | tee results/compare_hcc.txt
echo "=== [2/8] accuracy + ablations (MIMIC) ==="
python -m experiments.run           --dataset mimic --ablations 2>&1 | tee results/compare_mimic.txt
echo "=== [3/8] significance TOST/CI (HCC) ==="
python -m experiments.significance  --dataset hcc   2>&1 | tee results/sig_hcc.txt
echo "=== [4/8] significance TOST/CI (MIMIC) ==="
python -m experiments.significance  --dataset mimic 2>&1 | tee results/sig_mimic.txt
echo "=== [5/8] component stability (MIMIC) ==="
python -m experiments.ablations_extra --dataset mimic 2>&1 | tee results/ablextra_mimic.txt
echo "=== [6/8] protocol bench: comms/wall/PSI/routeA-B/convergence (MIMIC) ==="
python -m experiments.bench_protocol  --dataset mimic 2>&1 | tee results/bench_mimic.txt
echo "=== [7/8] DP epsilon accounting (MIMIC) ==="
python -m experiments.dp_curve        --dataset mimic 2>&1 | tee results/dp_mimic.txt
echo "=== [8/8] figures ==="
python -m experiments.make_figures 2>&1 | tee results/figures.txt

date; echo "##### analysis complete #####"
echo "all generated macros:"; grep -h "renewcommand" results/*.txt | sort -u
