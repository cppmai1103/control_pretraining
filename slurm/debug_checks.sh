#!/bin/bash
# Gate 4 launch checks on disposable models (plan Sections 11 and 13).
#   sbatch slurm/debug_checks.sh                 # 1 GPU
#   sbatch --gres=gpu:2 slurm/debug_checks.sh    # 2 GPUs (smoke + resume use both)
# Results: results/debug/*  -- read the [GATE4 ...] lines and the throughput/memory/checkpoint size.

#SBATCH --job-name=gpt2-debug
#SBATCH --output=logs/debug-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=03:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
SRC=src/pretrain_gpt2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
NGPUS=$(nvidia-smi -L | wc -l)
RUN="$PY -m torch.distributed.run --standalone --nproc_per_node=$NGPUS"
hostname; nvidia-smi; echo "GPUs: $NGPUS"
status=0

echo "=== 1. Tiny-batch overfitting (one GPU) ==="
rm -rf results/debug/overfit
$PY $SRC/train.py --overfit-tiny --out-dir results/debug/overfit || status=1

echo "=== 2. Smoke test: 200 steps, initial loss, LR schedule, eval, checkpoints, throughput ==="
rm -rf results/debug/smoke
$RUN $SRC/train.py --out-dir results/debug/smoke --max-steps 200 \
    --set eval.quick_interval=100 --set eval.full_interval=200 --set checkpoint.interval=100 || status=1
grep -h '"initial_loss"' results/debug/smoke/events.jsonl || true
tail -n 1 results/debug/smoke/train_log.jsonl || true
ls -la results/debug/smoke/checkpoints || true

echo "=== 3. Resume test: 20 steps continuous vs 10 + restart + 10 ==="
$PY $SRC/check_resume.py --nproc "$NGPUS" --base-dir results/debug/resume_test || status=1

echo "=== Gate 4 summary (exit status $status; 0 = all automatic checks passed) ==="
cat results/debug/overfit/gate4_overfit.json 2>/dev/null | head -4
cat results/debug/resume_test/gate4_resume.json 2>/dev/null | head -6
exit $status
