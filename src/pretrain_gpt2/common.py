"""Shared helpers for the five-language GPT-2 pipeline.

Every script imports from here so that seeds, file layouts, the loss, the
learning-rate schedule, and evaluation are defined in exactly one place.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterator

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply `key.sub=value` overrides (values parsed as YAML)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            try:  # PyYAML reads "3e-3" (no decimal point) as a string
                value = float(value)
            except ValueError:
                pass
        node[parts[-1]] = value
    return cfg


def add_override_arg(parser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.micro_batch=4 (repeatable)",
    )


# ---------------------------------------------------------------------------
# Seeds: one independent generator per data stage (plan, Section 2)
# ---------------------------------------------------------------------------


def stage_rng(seed: int) -> random.Random:
    """A private Python RNG for one data stage. Never use the global `random`."""
    return random.Random(seed)


def split_for_group(group_id: str, seed: int, train_frac: float, val_frac: float) -> str:
    """Deterministic split from the duplicate-group ID only (plan, Section 5).

    Documents collected later always land in the same split as before.
    """
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | os.PathLike, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def line_offsets(path: str | os.PathLike) -> list[int]:
    """Byte offset of every non-empty line, for seeded random access to large JSONL files."""
    offsets = []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            if line.strip():
                offsets.append(pos)
            pos += len(line)
    return offsets


def read_jsonl_at(f, offset: int) -> dict:
    f.seek(offset)
    return json.loads(f.readline())


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------


def clean_path(data_root: str, lang: str, split: str) -> Path:
    return Path(data_root) / "clean" / lang / f"{split}.jsonl"


def packed_path(data_root_or_packed: str, split: str, lang: str) -> Path:
    return Path(data_root_or_packed) / f"{split}_{lang}.bin"


def load_packed(packed_dir: str, split: str, lang: str, block_size: int, n_blocks: int | None = None,
                in_memory: bool = True):
    """Return a (n_blocks, block_size) uint16 array of packed token blocks."""
    import numpy as np

    path = packed_path(packed_dir, split, lang)
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    if arr.size % block_size:
        raise ValueError(f"{path} size is not a multiple of block size {block_size}")
    arr = arr.reshape(-1, block_size)
    if n_blocks is not None:
        if arr.shape[0] < n_blocks:
            raise ValueError(f"{path} has {arr.shape[0]} blocks, need {n_blocks}")
        arr = arr[:n_blocks]
    return np.array(arr) if in_memory else arr


# ---------------------------------------------------------------------------
# Model, loss, schedule, evaluation (shared by train.py and evaluate.py)
# ---------------------------------------------------------------------------


def build_model(model_cfg: dict, vocab_size: int, eos_id: int, pad_id: int):
    """GPT-2 config from the plan (Section 8). BOS = EOS, following GPT-2."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=model_cfg["n_positions"],
        n_embd=model_cfg["n_embd"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_inner=model_cfg["n_inner"],
        activation_function=model_cfg["activation_function"],
        resid_pdrop=model_cfg["resid_pdrop"],
        embd_pdrop=model_cfg["embd_pdrop"],
        attn_pdrop=model_cfg["attn_pdrop"],
        bos_token_id=eos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config._attn_implementation = model_cfg.get("attn_implementation", "sdpa")
    return GPT2LMHeadModel(config)


def lm_loss_sum(model, input_ids):
    """Summed next-token cross-entropy and number of targets for full blocks.

    Labels are the unshifted inputs; the shift happens here exactly once.
    Summing (not averaging) lets the caller normalize by the true number of
    targets in the whole optimizer step, across microbatches and GPUs.
    """
    import torch.nn.functional as F

    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), targets.reshape(-1), reduction="sum"
    )
    return loss, targets.numel()


def lr_at(step: int, peak: float, min_ratio: float, warmup: int, total: int) -> float:
    """Learning rate for optimizer step `step` (1-based), plan Section 10.

    Linear warmup reaching the peak at step `warmup`, then cosine decay to
    `peak * min_ratio` exactly at step `total`. Equivalent to Hugging Face's
    get_cosine_with_min_lr_schedule_with_warmup(min_lr_rate=min_ratio).
    """
    if step <= 0:
        return 0.0
    if step <= warmup:
        return peak * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    min_lr = peak * min_ratio
    return min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate_blocks(model, arrays: dict, micro_batch: int, device, autocast_ctx, rank: int = 0,
                    world: int = 1) -> dict:
    """Loss-only, per-language evaluation (plan, Section 12).

    `arrays` maps language -> (n, block) uint16 array. Rows are sharded across
    ranks; loss sums and target counts are all-reduced, never full logits.
    Returns per-language loss/perplexity plus macro (equal-weight) and micro
    (token-weighted) averages.
    """
    import numpy as np
    import torch
    import torch.distributed as dist

    langs = list(arrays)
    sums = torch.zeros(len(langs), 2, dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, lang in enumerate(langs):
            rows = arrays[lang][rank::world]
            for start in range(0, len(rows), micro_batch):
                x = torch.from_numpy(rows[start:start + micro_batch].astype(np.int64)).to(device)
                with autocast_ctx():
                    loss, n = lm_loss_sum(model, x)
                sums[i, 0] += loss.double()
                sums[i, 1] += n
    if world > 1:
        dist.all_reduce(sums)
    if was_training:
        model.train()
    out = {"per_language": {}}
    for i, lang in enumerate(langs):
        mean = (sums[i, 0] / sums[i, 1]).item()
        out["per_language"][lang] = {
            "loss": mean,
            "ppl": math.exp(min(mean, 50.0)),
            "targets": int(sums[i, 1].item()),
        }
    losses = [v["loss"] for v in out["per_language"].values()]
    out["macro_loss"] = sum(losses) / len(losses)
    out["micro_loss"] = (sums[:, 0].sum() / sums[:, 1].sum()).item()
    return out


def repetition_score(token_ids: list[int], n: int = 4) -> float:
    """Fraction of distinct n-grams; low values mean degenerate repetition."""
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0
