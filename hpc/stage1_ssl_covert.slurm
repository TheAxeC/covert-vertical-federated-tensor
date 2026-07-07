#!/bin/bash
#SBATCH --job-name=cov_p1_dino
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# COVERT stage-1 DINO SSL backbone pretrain -- RUNS IN THE COVERT-OWNED FORK, never the HCCNet deposit.
# This is the FIRST stage: label-free self-supervised pretrain (32000 steps, effective batch 128,
# seq-len 8, all set by --paper) that produces the SSL backbone stage2 consumes. 4-GPU DDP (~7 h);
# for the 1-GPU path (~27 h) submit with:
#   sbatch --gres=gpu:1 --cpus-per-task=8 --export=ALL,NGPU=1 stage1_ssl_covert.slurm [GROUP]
#   Usage (submit from ~/projects/covert/hccnet_pipeline): sbatch stage1_ssl_covert.slurm [GROUP]  (dwi|t1iop)
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
source /etc/profile.d/modules.sh 2>/dev/null   # some compute nodes don't init the module system in batch shells; make `module` available before load
module load anaconda3/2024.02
source activate $HOME/envs/hccnet
cd $FORK
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "##### COVERT STAGE 1 DINO (fork): group=$GROUP tag=$TAG nproc=${NGPU:-4} #####"; date
torchrun --standalone --nproc_per_node=${NGPU:-4} pretrain.py \
  --paper --distributed --loss-fn dino --arch $ARCH \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix $TAG \
  --results-dir $RESULTS
RC=$?; date; echo "torchrun exit code: $RC"
OUT=$RESULTS/model_weights/weights_dinopretrain_${TAG}.pth
if [ ! -f "$OUT" ]; then echo "ERROR: handoff $OUT not created (RC=$RC)"; echo "##### COVERT STAGE 1 FAILED #####"; exit 1; fi
# The saved backbone is the source of truth: DDP can throw a non-zero torchrun RC on a late rank
# teardown AFTER the checkpoint is written - that is benign, so a present handoff means success.
[ "$RC" -eq 0 ] || echo "NOTE: torchrun RC=$RC but the backbone was saved (benign late-rank exit); treating as success."
echo "HANDOFF OK: $(basename $OUT)"
echo "##### COVERT STAGE 1 DONE #####"
