#!/bin/bash
#SBATCH --job-name=cov_p2_decoder
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#
# COVERT stage-2 decoder pretrain — RUNS IN THE COVERT-OWNED FORK, never the HCCNet deposit.
# Label-free SSL (sequence-order); excludes the held-out test patients via the controlled split.
# Reads/writes COVERT results dir; loads the SSL backbone seeded there.
#   Usage (submit from ~/projects/covert/hccnet_pipeline): sbatch stage2_decoder_covert.slurm [GROUP]  (dwi|t1iop)
#   Canonical copy: covert-applied/code/hpc/ (rsync/scp to the fork before submitting).
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
export COVERT_LABEL_FILE=labels_12months.csv     # (pretrain phase uses lirads.csv; harmless here)
source /etc/profile.d/modules.sh 2>/dev/null   # some compute nodes do not init the module system in batch shells; make `module` available
module load anaconda3/2024.02
source activate $HOME/envs/hccnet
cd $FORK
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
NEED=$RESULTS/model_weights/weights_dinopretrain_${TAG}.pth
if [ ! -f "$NEED" ]; then echo "ABORT: missing SSL backbone $NEED"; exit 1; fi
echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "##### COVERT STAGE 2 DECODER (fork): group=$GROUP tag=$TAG split=$COVERT_SPLIT_FILE #####"; date
torchrun --standalone --nproc_per_node=1 pretrain.py \
  --paper --distributed --loss-fn bce --arch $ARCH \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix ${TAG}_te \
  --results-dir $RESULTS
RC=$?; date; echo "torchrun exit code: $RC"
OUT=$RESULTS/model_weights/weights_decoderpretrain_${TAG}.pth
if [ -f "$OUT" ]; then echo "HANDOFF OK: $(basename $OUT)"; else echo "ERROR: handoff $OUT not created"; fi
echo "##### COVERT STAGE 2 DONE #####"
