#!/bin/bash
# CPU job for data preparation. Pass one stage:
#   sbatch slurm/prepare.sh data        # collect + clean + split (Gate 1 automatic checks)
#   sbatch slurm/prepare.sh tokenizer   # train tokenizer + Gate 2 report
#   sbatch slurm/prepare.sh pack        # tokenize + pack blocks + Gate 3
# Stop between stages to review the reports in data/reports/.

#SBATCH --job-name=gpt2-prep
#SBATCH --output=logs/prep-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
SRC=src/pretrain_gpt2
[ -f .env ] && { set -a; . ./.env; set +a; }     # HF_TOKEN (FLORES+ is gated)
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}  # tokenizers' thread pool

STAGE=${1:?usage: sbatch slurm/prepare.sh data|tokenizer|pack}
hostname
case "$STAGE" in
  data)      $PY $SRC/prepare_data.py --stage all ;;
  tokenizer) $PY $SRC/train_tokenizer.py && $PY $SRC/inspect_tokenizer.py ;;
  pack)      $PY $SRC/pack_tokens.py ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
