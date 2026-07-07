#!/bin/bash
#SBATCH --job-name=cov_embed_ft
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#
# COVERT OOF leak-free fine-tuned-backbone embedding extraction as a PROPER SLURM job
# (not a login-node process). Forwards run on the GPU; the MONAI transform stays CPU.
# Submit from the package root ~/projects/covert:  sbatch code/hpc/extract_embeddings_covert.slurm
source /etc/profile.d/modules.sh 2>/dev/null   # some compute nodes do not init the module system in batch shells; make `module` available
module load anaconda3/2024.02
source activate $HOME/envs/hccnet
cd $HOME/projects/covert
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "##### COVERT OOF FT EMBEDDING EXTRACTION #####"; date
COVERT_HCC_FEATURES=$HOME/projects/covert/hcc_embeddings.npz \
  python -u -m experiments.extract_hcc_embeddings_ft --device cuda
RC=$?; date; echo "exit code: $RC"
ls -la $HOME/projects/covert/hcc_embeddings.npz 2>&1
echo "##### DONE #####"
