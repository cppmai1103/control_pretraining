# GPT-2 from scratch: five languages, 500M tokens

Code for the plan in [docs/GPT2_500M_5_language_training_plan.md](docs/GPT2_500M_5_language_training_plan.md):
a GPT-2 Small–style model (about 135M parameters, 64,000-token vocabulary) trained from random
initialization on English, Vietnamese, Indonesian, Japanese and Chinese, with exactly 97,656 blocks
of 1,024 tokens per language (499,998,720 tokens in total). Training runs on 1 or 2 GPUs with the same
effective batch of 128 sequences.

## Files

| File | Purpose |
|---|---|
| `configs/data.yaml` | Sources, sizes, cleaning thresholds, split, tokenizer, packing, Gate 1–3 thresholds |
| `configs/train.yaml` | Model, optimizer, batch, evaluation, checkpoints, Gate 4–5 thresholds |
| `src/pretrain_gpt2/prepare_data.py` | Collect, clean, deduplicate, split (Gate 1) |
| `src/pretrain_gpt2/train_tokenizer.py` | Train the shared byte-level BPE tokenizer |
| `src/pretrain_gpt2/inspect_tokenizer.py` | Gate 2 tokenizer checks and report |
| `src/pretrain_gpt2/pack_tokens.py` | Monolingual 1,024-token blocks and manifests (Gate 3) |
| `src/pretrain_gpt2/train.py` | Training, evaluation, generation, checkpoints, resume, Gate 4/5 checks |
| `src/pretrain_gpt2/check_resume.py` | Gate 4 resume test |
| `src/pretrain_gpt2/evaluate.py` | Per-language validation/test loss and generation for a saved model |
| `src/pretrain_gpt2/common.py` | Shared helpers: seeds, split hash, model, loss, LR schedule, evaluation |
| `slurm/*.sh` | Cluster jobs: data preparation, debug checks, training |

## Setup (once)

```bash
conda create -y -p ./.venv python=3.11
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt     # exact versions: environment-lock.txt
mkdir -p logs
```

`.env` must export `HF_TOKEN`, and that account must have accepted the terms of the gated
[FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) dataset (used by Gate 2).

Run the heavy steps as Slurm jobs, not on the login node: it kills long CPU-heavy processes.

## Workflow

Each step must pass its gate (plan Section 13) before the next starts.

**1. Data** — collect, clean and split:

```bash
sbatch slurm/prepare.sh data
```

Gate 1: the job fails if any document or duplicate group crosses splits. Then read
`data/reports/inspect/<lang>_kept.jsonl` and `<lang>_rejected.jsonl` by hand: at least 95% of kept
documents should be mostly in the right language. Adjust `cleaning` in `configs/data.yaml` and rerun
`src/pretrain_gpt2/prepare_data.py --stage clean` if needed. Summaries: `data/reports/cleaning_<lang>.json`.

Collection streams data from the Hugging Face Hub, so the compute node needs internet access.
Dataset revisions are resolved to commit hashes and recorded in `data/raw/<lang>.meta.json`.

**2. Tokenizer** — train it and run Gate 2:

```bash
sbatch slurm/prepare.sh tokenizer
```

Read `data/reports/gate2_tokenizer.md`. If it fails, change `tokenizer.sample_bytes_per_language`
(or give the weak language more text), retrain, and rerun. Freeze the tokenizer only after a pass.

**3. Packing** — tokenize and build blocks, then Gate 3:

```bash
sbatch slurm/prepare.sh pack
```

The job fails if any language lacks text for 97,656 training blocks or 1,024 validation/test
blocks (that is the "at least 102M tokens" part of Gate 1). Raise `target_bytes` for that
language, recollect, and repack. Then read `data/reports/gate3_decoded_blocks.txt`.

**4. Debug checks (Gate 4)** — disposable models:

```bash
sbatch slurm/debug_checks.sh                 # or: sbatch --gres=gpu:2 slurm/debug_checks.sh
```

This runs tiny-batch overfitting, a 200-step smoke test (initial loss about 11.07, LR schedule,
evaluation, checkpoints) and the resume test. The job exits non-zero if an automatic check fails. Note
the throughput, peak memory and checkpoint size from `logs/debug-<id>.out`. If memory runs out, use
`--set train.micro_batch=4` (the effective batch stays 128).

**5. Training** — fresh initialization, monitored by Gate 5:

```bash
sbatch slurm/train.sh                        # 1 GPU
sbatch --gres=gpu:2 slurm/train.sh           # 2 GPUs
```

If the job ends early (time limit, node failure), submit the same command again: it resumes from the
latest checkpoint in `results/gpt2_5lang_500M/checkpoints/`. Gate 5 warnings are printed and stored
in `events.jsonl`. Exit code 3 (non-finite loss) or 4 (loss spike without recovery) means the run
stopped; resume from the last checkpoint only after fixing the cause, and record the change.

**6. Evaluation**

```bash
.venv/bin/python src/pretrain_gpt2/evaluate.py --model results/gpt2_5lang_500M/final_model --split val --generate
.venv/bin/python src/pretrain_gpt2/evaluate.py --model results/gpt2_5lang_500M/final_model --split test --confirm-test
```

Use the test split once, after model selection.

## Outputs

`results/gpt2_5lang_500M/`:

| Path | Contents |
|---|---|
| `train_log.jsonl` | Loss, LR, gradient norm, tokens/s, per-language token share, memory (every 10 steps) |
| `eval_log.jsonl` | Per-language validation loss and perplexity, macro and token-weighted averages |
| `events.jsonl` | Gate 4/5 pass, warning and stop events |
| `generations.jsonl` | Fixed-prompt samples at steps 0, 1000, 2000, 3000 and the end |
| `checkpoints/step_*.pt` | Full resumable checkpoints every 500 steps: model, optimizer, RNG per GPU, data position, monitor state. Keeps the last 2, plus steps 1,000, 2,000, 3,000 and the final step permanently (about 1.6GB each) |
| `snapshots/step_*/` | Weights-only Hugging Face models for analysis at steps 0, 16, 32, 64, 128, every 500, and 3,815 (13 snapshots, about 0.54GB each, never deleted). `snapshot_info.json` gives the tokens seen; `snapshots/tokenizer/` holds the shared tokenizer |
| `final_model/` | Hugging Face model and tokenizer |
| `summary.json` | Tokens processed per language, GPU hours, throughput, peak memory |
| `run_config.json` | Full resolved configuration, GPU count, parameter count, versions |

## Implementation notes

- **1 vs 2 GPUs.** Each optimizer step takes the next 128 blocks from one fixed order (seed 42) and
  splits them across GPUs; gradient accumulation is `128 / (micro_batch × GPUs)`. The loss is summed
  and divided by the number of targets in the whole step, so the last 88-sequence step and uneven
  splits are handled exactly. With dropout off, 1- and 2-process CPU runs gave the same losses to about
  1e-5 per step (bf16 rounding).
- **Resume** restores model, optimizer, step, data position, per-GPU RNG and monitor state. On CPU,
  the resume test matched the continuous run exactly for both 1 and 2 processes. Resuming with a
  different GPU count keeps the data order but changes the dropout random streams (a warning is logged).
- **Learning rate.** Linear warmup to 3e-4 at step 77, cosine decay to exactly 3e-5 at step 3,815
  (equivalent to `get_cosine_with_min_lr_schedule_with_warmup(min_lr_rate=0.1)`).
- **Special tokens.** `<pad>` = 0 and `<eos>` = 1, and `<eos>` is also BOS. Literal `<eos>` text in
  documents is encoded as ordinary characters (`encode_special_tokens = True`).
- **Overrides.** Any config value can be changed with `--set key.sub=value`, for example
  `--set train.micro_batch=4`.
