#!/bin/bash
#SBATCH --job-name=cov_mimic_feat
#SBATCH --partition=main-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# CPU-only (scipy Welch + pandas; no GPU) -> no --gres so it backfills onto free CPU cores fast.
#
# COVERT MIMIC-IV 3-party feature extraction (ecg multiway 12x8 + labs + vitals, 1-yr mortality).
# Reads the cluster ~/data MIMIC-IV-ECG zip + MIMIC-IV hosp CSVs; writes mimic_features.npz.
# Submit from the package root ~/projects/covert:  sbatch hpc/extract_mimic_covert.slurm [COHORT_CAP]
# Canonical copy: covert-applied/code/hpc/ (rsync/scp to the cluster before submitting).
source /etc/profile.d/modules.sh 2>/dev/null   # some compute nodes do not init the module system in batch shells; make `module` available
module load anaconda3/2024.02
source activate $HOME/envs/hccnet          # NB needs numpy<2 (wfdb pulls numpy 2.x, which breaks torch)
cd $HOME/projects/covert
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "##### COVERT MIMIC feature extraction (cohort cap=${1:-5000}) #####"; date
COVERT_MIMIC_FEATURES=$HOME/data/mimic-iv/covert/mimic_features.npz \
  python -m experiments.extract_mimic_features --limit ${1:-5000} --workers 8
RC=$?; date; echo "exit code: $RC"
