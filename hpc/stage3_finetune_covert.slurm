#!/bin/bash
#SBATCH --job-name=cov_p3_finetune
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#
# COVERT stage-3 finetune — RUNS IN THE COVERT-OWNED FORK, never the HCCNet deposit.
# Supervised finetune on the incident-12mo outcome, ONE backbone per CV fold using the
# CONTROLLED saved split (num_seeds patched to 1 in paper_config -> 5 runs = folds 0..4).
# Outputs weights_fold{0..4}_<tag>_ft.pth under the COVERT results dir (leak-free: fold k trained
# only on the split's fold-k train patients).
#   Usage (submit from ~/projects/covert/hccnet_pipeline): sbatch stage3_finetune_covert.slurm [GROUP]  (dwi|t1iop)
#   Canonical copy: covert-applied/code/hpc/.
GROUP="${1:-dwi}"; ARCH="femto"
case "$GROUP" in
  dwi)   MODLIST="DWI_b0 DWI_b150 DWI_b400 DWI_b800" ;;
  t1iop) MODLIST="T1W_IP T1W_OOP T2W_TES T2W_TEL" ;;
  *) echo "unknown GROUP '$GROUP' (use dwi|t1iop)"; exit 1 ;;
esac
TAG="${GROUP}_${ARCH}_paper"
FORK=$HOME/projects/covert/hccnet_pipeline
RESULTS=$HOME/projects/covert/results
export COVERT_SPLIT_FILE=$HOME/projects/covert/splits/hcc_split_h12_seed1234.json
export COVERT_LABEL_FILE=labels_12months.csv     # the incident-12mo outcome (COVERT's task)
source /etc/profile.d/modules.sh 2>/dev/null   # some compute nodes do not init the module system in batch shells; make `module` available
module load anaconda3/2024.02
source activate $HOME/envs/hccnet
cd $FORK
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
NEED=$RESULTS/model_weights/weights_decoderpretrain_${TAG}.pth
if [ ! -f "$NEED" ]; then echo "ABORT: missing decoder hand-off $NEED"; exit 1; fi
echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "##### COVERT STAGE 3 FINETUNE (fork): group=$GROUP tag=$TAG label=$COVERT_LABEL_FILE split=$COVERT_SPLIT_FILE #####"; date
torchrun --standalone --nproc_per_node=1 train.py \
  --paper --distributed --pretrained --arch $ARCH \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix ${TAG}_ft \
  --results-dir $RESULTS
RC=$?; date; echo "torchrun exit code: $RC"
echo "fold checkpoints:"; ls $RESULTS/model_weights/weights_fold*_${TAG}_ft.pth 2>/dev/null
echo "##### COVERT STAGE 3 DONE #####"
