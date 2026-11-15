#!/bin/bash
# Final training run (plan Sections 9-13). Resubmitting resumes from the latest checkpoint.
#   sbatch slurm/train.sh                    # 1 GPU
#   sbatch --gres=gpu:2 slurm/train.sh       # 2 GPUs, same effective batch (128)
# Exit codes: 3 = non-finite loss (Gate 5 STOP), 4 = unrecovered loss spike (Gate 5 STOP).

#SBATCH --job-name=gpt2-5lang
#SBATCH --output=logs/train-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --time=24:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
SRC=src/pretrain_gpt2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
NGPUS=$(nvidia-smi -L | wc -l)
hostname; nvidia-smi; echo "GPUs: $NGPUS"

$PY -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" \
    $SRC/train.py --config configs/train.yaml --resume auto "$@"
code=$?
case $code in
  0) echo "training finished or reached --max-steps" ;;
  3) echo "Gate 5 STOP: non-finite loss. See results/*/events.jsonl" ;;
  4) echo "Gate 5 STOP: loss spike did not recover. See results/*/events.jsonl" ;;
  *) echo "training exited with code $code" ;;
esac
exit $code
