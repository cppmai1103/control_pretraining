"""Train the five-language GPT-2 baseline from scratch (plan Sections 8-13).

Single GPU:   python src/pretrain_gpt2/train.py
Two GPUs:     torchrun --standalone --nproc_per_node=2 src/pretrain_gpt2/train.py
Resume:       add --resume auto   (latest checkpoint in out_dir)
Smoke test:   python src/pretrain_gpt2/train.py --out-dir results/debug/smoke --max-steps 200
Tiny overfit: python src/pretrain_gpt2/train.py --out-dir results/debug/overfit --overfit-tiny   (Gate 4)

The effective batch is always `global_batch` sequences (128): each optimizer
step takes the next 128 blocks of one fixed, seeded order of all 488,280
blocks, splits them across GPUs, and accumulates microbatches. The loss is
summed and divided by the true number of targets in the whole step, so the
final 88-sequence step and uneven GPU shares are normalized correctly.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import shutil
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from common import (add_override_arg, append_jsonl, build_model, ensure_dir, evaluate_blocks, load_config,
                    load_packed, lm_loss_sum, lr_at, read_json, repetition_score, write_json)

EXIT_NAN = 3
EXIT_SPIKE = 4


# ---------------------------------------------------------------------------
# Distributed setup and RNG state
# ---------------------------------------------------------------------------


def setup_distributed(device_arg: str):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg == "cpu" or not torch.cuda.is_available():
        device, backend = torch.device("cpu"), "gloo"
    else:
        torch.cuda.set_device(local)
        device, backend = torch.device("cuda", local), "nccl"
    if world > 1:
        dist.init_process_group(backend)
    return rank, world, local, device


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def rng_state(device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def set_rng_state(state: dict, device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def gather_rng(world: int, device) -> list:
    state = rng_state(device)
    if world == 1:
        return [state]
    out = [None] * world
    dist.all_gather_object(out, state)
    return out


# ---------------------------------------------------------------------------
# Logging and Gate 5 health monitor
# ---------------------------------------------------------------------------


class RunLog:
    def __init__(self, out_dir: Path, is_main: bool):
        self.is_main = is_main
        self.train_path = out_dir / "train_log.jsonl"
        self.eval_path = out_dir / "eval_log.jsonl"
        self.event_path = out_dir / "events.jsonl"
        self.gen_path = out_dir / "generations.jsonl"

    def info(self, msg: str) -> None:
        if self.is_main:
            print(msg, flush=True)

    def write(self, kind: str, record: dict) -> None:
        if self.is_main:
            append_jsonl({"train": self.train_path, "eval": self.eval_path,
                          "gen": self.gen_path}[kind], record)

    def event(self, step: int, level: str, gate: str, check: str, message: str) -> None:
        if self.is_main:
            append_jsonl(self.event_path, {"step": step, "level": level, "gate": gate, "check": check,
                                           "message": message, "time": time.time()})
            print(f"[{gate.upper()} {level}] step {step} {check}: {message}", flush=True)


class HealthMonitor:
    """Gate 5 warning and stop rules (plan Section 13). All inputs are global
    (identical on every rank), so every rank reaches the same decision."""

    def __init__(self, g: dict, langs: list[str]):
        self.g, self.langs = g, langs
        self.ema = None
        self.ema_min = math.inf
        self.spike_steps = 0
        self.grad_hist = deque(maxlen=200)
        self.clipped = deque(maxlen=100)
        self.last_grad_warn = -10**9
        self.last_clip_warn = -10**9
        self.best_val: dict[str, float] = {}
        self.no_improve = {l: 0 for l in langs}
        self.recent_train = deque(maxlen=50)
        self.tp_early: list[float] = []
        self.tp_baseline = None
        self.last_tp_warn = -10**9
        self.last_share_warn = -10**9

    def state_dict(self) -> dict:
        d = {k: (list(v) if isinstance(v, deque) else v) for k, v in self.__dict__.items() if k != "g"}
        return d

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            cur = getattr(self, k, None)
            setattr(self, k, deque(v, maxlen=cur.maxlen) if isinstance(cur, deque) else v)

    def on_step(self, step: int, loss: float, grad_norm: float, clip: float, log: RunLog) -> bool:
        g = self.g
        self.recent_train.append(loss)
        a = g["loss_ema_alpha"]
        self.ema = loss if self.ema is None else a * loss + (1 - a) * self.ema
        self.ema_min = min(self.ema_min, self.ema)
        if self.ema > g["loss_spike_ratio"] * self.ema_min:
            self.spike_steps += 1
            if self.spike_steps == g["loss_spike_warn_steps"]:
                log.event(step, "WARN", "gate5", "loss_spike",
                          f"smoothed loss {self.ema:.4f} > {g['loss_spike_ratio']}x minimum {self.ema_min:.4f} "
                          f"for {self.spike_steps} steps")
            if self.spike_steps >= g["loss_spike_stop_steps"]:
                log.event(step, "STOP", "gate5", "loss_spike",
                          f"no recovery after {self.spike_steps} steps: restart from the last checkpoint "
                          f"with a lower learning rate and record the change")
                return True
        else:
            self.spike_steps = 0

        if len(self.grad_hist) >= 20:
            med = statistics.median(self.grad_hist)
            if grad_norm > g["grad_spike_ratio"] * med and step - self.last_grad_warn >= 50:
                log.event(step, "WARN", "gate5", "grad_spike",
                          f"pre-clip grad norm {grad_norm:.3f} > {g['grad_spike_ratio']}x median {med:.3f}")
                self.last_grad_warn = step
        self.grad_hist.append(grad_norm)
        self.clipped.append(grad_norm > clip)
        if len(self.clipped) == self.clipped.maxlen and step - self.last_clip_warn >= 100:
            frac = sum(self.clipped) / len(self.clipped)
            if frac > g["grad_clip_warn_fraction"]:
                log.event(step, "WARN", "gate5", "grad_clipping",
                          f"clipped on {frac:.0%} of the last 100 steps; learning rate may be too high")
                self.last_clip_warn = step
        return False

    def on_lang_share(self, step: int, lang_tokens: np.ndarray, log: RunLog) -> None:
        if step < self.g["lang_share_check_after"] or step - self.last_share_warn < 100:
            return
        share = lang_tokens / lang_tokens.sum()
        expected = 1.0 / len(self.langs)
        bad = {l: round(float(s), 4) for l, s in zip(self.langs, share)
               if abs(s - expected) > self.g["lang_share_tolerance"]}
        if bad:
            log.event(step, "WARN", "gate5", "language_share",
                      f"cumulative shares {bad} drift > {self.g['lang_share_tolerance']} from {expected:.2f}")
            self.last_share_warn = step

    def on_throughput(self, step: int, tps: float, log: RunLog) -> None:
        if self.tp_baseline is None:
            if 20 <= step <= 120:
                self.tp_early.append(tps)
            elif step > 120 and self.tp_early:
                self.tp_baseline = statistics.median(self.tp_early)
                log.info(f"[gate5] throughput baseline {self.tp_baseline:,.0f} tokens/s")
            return
        if tps < (1 - self.g["throughput_drop"]) * self.tp_baseline and step - self.last_tp_warn >= 100:
            log.event(step, "WARN", "gate5", "throughput",
                      f"{tps:,.0f} tokens/s is >{self.g['throughput_drop']:.0%} below baseline "
                      f"{self.tp_baseline:,.0f}; check GPU, data loading, storage")
            self.last_tp_warn = step

    def on_full_eval(self, step: int, res: dict, log: RunLog) -> None:
        for lang, v in res["per_language"].items():
            best = self.best_val.get(lang)
            if best is None or v["loss"] < best:
                self.best_val[lang] = v["loss"]
                self.no_improve[lang] = 0
            else:
                self.no_improve[lang] += 1
                if self.no_improve[lang] >= self.g["val_no_improve_evals"]:
                    log.event(step, "WARN", "gate5", "val_not_improving",
                              f"{lang} validation loss has not improved for {self.no_improve[lang]} full "
                              f"evaluations (best {best:.4f}, now {v['loss']:.4f}); inspect {lang} data")
        if self.recent_train:
            train = statistics.mean(self.recent_train)
            gap = res["micro_loss"] - train
            if gap > self.g["val_train_gap"]:
                log.event(step, "WARN", "gate5", "val_train_gap",
                          f"validation {res['micro_loss']:.4f} vs recent train {train:.4f} (gap {gap:.3f}); "
                          f"held-out data may be mismatched or broken")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def checkpoint_files(ckpt_dir: Path) -> list[Path]:
    return sorted(ckpt_dir.glob("step_*.pt"))


def save_checkpoint(ckpt_dir: Path, step: int, total_steps: int, raw_model, optimizer, extra: dict,
                    ccfg: dict, is_main: bool) -> None:
    if not is_main:
        return
    path = ckpt_dir / f"step_{step:06d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, **extra}, tmp)
    os.replace(tmp, path)
    keep_last = int(ccfg["keep_last"])
    milestone = int(ccfg["milestone_interval"])
    files = checkpoint_files(ckpt_dir)
    recent = set(files[-keep_last:])
    for f in files:
        s = int(f.stem.split("_")[1])
        if f in recent or s % milestone == 0 or s == total_steps:
            continue
        f.unlink()
    size = path.stat().st_size / 1e9
    print(f"[ckpt] saved {path} ({size:.2f} GB)", flush=True)


def is_snapshot_step(step: int, total_steps: int, ccfg: dict) -> bool:
    interval = int(ccfg.get("snapshot_interval") or 0)
    return (step in ccfg.get("snapshot_steps", []) or step == total_steps
            or (interval > 0 and step > 0 and step % interval == 0))


def save_snapshot(out_dir: Path, step: int, raw_model, tokenizer_dir: Path, lang_tokens: np.ndarray,
                  langs: list[str], is_main: bool) -> None:
    """Weights-only snapshot for later analysis; kept permanently, never overwritten."""
    if not is_main:
        return
    snap_root = ensure_dir(out_dir / "snapshots")
    tok_copy = snap_root / "tokenizer"
    if not tok_copy.exists():   # one shared tokenizer copy for all snapshots
        tmp_tok = snap_root / "tokenizer.tmp"
        shutil.rmtree(tmp_tok, ignore_errors=True)
        ensure_dir(tmp_tok)
        for f in tokenizer_dir.iterdir():
            if f.is_file() and f.suffix == ".json":
                shutil.copy2(f, tmp_tok / f.name)
        os.replace(tmp_tok, tok_copy)
    final = snap_root / f"step_{step:06d}"
    if final.exists():          # already saved before a restart
        return
    tmp = snap_root / f"step_{step:06d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    raw_model.save_pretrained(tmp)
    write_json(tmp / "snapshot_info.json", {
        "step": step,
        "input_tokens": int(lang_tokens.sum()),
        "tokens_per_language": {l: int(t) for l, t in zip(langs, lang_tokens)},
        "tokenizer": "../tokenizer",
    })
    os.replace(tmp, final)
    print(f"[snapshot] saved {final}", flush=True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_samples(raw_model, tok, ecfg: dict, eos_id: int, pad_id: int, device, autocast_ctx, step: int,
                     gcfg: dict, log: RunLog) -> None:
    saved = rng_state(device)
    torch.manual_seed(1234 + step)
    was_training = raw_model.training
    raw_model.eval()
    for lang, prompt in ecfg["prompts"].items():
        ids = [eos_id] + tok.encode(prompt).ids   # prompts start with <eos>, like document starts in training
        x = torch.tensor([ids], device=device)
        with torch.no_grad(), autocast_ctx():
            out = raw_model.generate(
                input_ids=x, attention_mask=torch.ones_like(x),
                max_new_tokens=int(ecfg["generation_max_new_tokens"]), do_sample=True,
                temperature=float(ecfg["generation_temperature"]), top_k=int(ecfg["generation_top_k"]),
                use_cache=True, pad_token_id=pad_id, eos_token_id=eos_id,
            )
        new = out[0, len(ids):].tolist()
        rep = repetition_score(new)
        text = tok.decode(new, skip_special_tokens=False)
        log.write("gen", {"step": step, "language": lang, "prompt": prompt, "completion": text,
                          "distinct_4gram_fraction": round(rep, 3)})
        log.info(f"[gen] step {step} {lang}: {prompt}{text[:160]!r}")
        if step >= 1000 and rep < gcfg["repetition_min_distinct_4grams"]:
            log.event(step, "WARN", "gate5", "generation_repetition",
                      f"{lang} completion repeats itself (distinct 4-grams {rep:.2f}); check {lang} data")
    if was_training:
        raw_model.train()
    set_rng_state(saved, device)   # generation must not change the training RNG stream


# ---------------------------------------------------------------------------
# Gate 4: tiny-batch overfitting
# ---------------------------------------------------------------------------


def run_overfit(raw_model, optimizer, train_arrays, cfg, device, autocast_ctx, out_dir: Path, log: RunLog) -> int:
    g, t = cfg["gates"], cfg["train"]
    n = int(g["overfit_blocks"])
    rows = [train_arrays[i % len(train_arrays)][i // len(train_arrays)] for i in range(n)]
    x = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
    raw_model.train()
    result = {"blocks": n, "target_loss": g["overfit_target_loss"], "passed_at_step": None, "losses": []}
    for k in range(1, int(g["overfit_max_steps"]) + 1):
        for group in optimizer.param_groups:
            group["lr"] = float(t["peak_lr"]) * min(1.0, k / 10)
        with autocast_ctx():
            loss_sum, n_t = lm_loss_sum(raw_model, x)
        loss = loss_sum / n_t
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), t["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        val = loss.item()
        result["losses"].append(round(val, 4))
        if k % 10 == 0 or k == 1:
            log.info(f"[overfit] step {k} loss {val:.4f}")
        if val < g["overfit_target_loss"]:
            result["passed_at_step"] = k
            break
    result["pass"] = result["passed_at_step"] is not None
    write_json(out_dir / "gate4_overfit.json", result)
    log.info(f"[gate4] tiny-batch overfitting {'PASS' if result['pass'] else 'FAIL'}: "
             f"final loss {result['losses'][-1]:.4f} (target < {g['overfit_target_loss']})")
    return 0 if result["pass"] else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--out-dir", default=None, help="override out_dir")
    parser.add_argument("--resume", default=None, help="'auto' (latest in out_dir) or a checkpoint path")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this optimizer step (schedule still uses the full length)")
    parser.add_argument("--no-eval", action="store_true", help="skip evaluation and generation")
    parser.add_argument("--overfit-tiny", action="store_true", help="Gate 4 tiny-batch overfitting check")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    add_override_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    rank, world, local, device = setup_distributed(args.device)
    is_main = rank == 0
    out_dir = Path(cfg["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    if is_main:
        ensure_dir(ckpt_dir)
    barrier(world)
    log = RunLog(out_dir, is_main)

    tcfg, mcfg, ecfg, gcfg = cfg["train"], cfg["model"], cfg["eval"], cfg["gates"]
    langs = list(cfg["languages"])
    L, B = len(langs), int(tcfg["block_size"])
    bpl = int(tcfg["blocks_per_language"])
    G, mb = int(tcfg["global_batch"]), int(tcfg["micro_batch"])

    # ---- data and tokenizer -------------------------------------------------
    data_dir = Path(cfg["data_dir"])
    meta = read_json(data_dir / "meta.json")
    if not meta.get("complete"):
        sys.exit("packed data is incomplete; rerun pack_tokens.py")
    if meta["languages"] != langs or meta["block_size"] != B:
        sys.exit(f"packed data languages/block size {meta['languages']}/{meta['block_size']} do not match config")
    if meta["train_blocks_per_language"] < bpl:
        sys.exit(f"packed data has {meta['train_blocks_per_language']} blocks/language, config needs {bpl}")
    vocab, eos_id, pad_id = int(meta["vocab_size"]), int(meta["eos_id"]), int(meta["pad_id"])

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(cfg["tokenizer_dir"]) / "tokenizer.json"))
    tok.encode_special_tokens = True
    if tok.get_vocab_size() != vocab or tok.token_to_id("<eos>") != eos_id:
        sys.exit("tokenizer does not match the one used for packing")

    train_arrays = [load_packed(data_dir, "train", l, B, bpl) for l in langs]
    N = L * bpl
    order = np.random.default_rng(int(cfg["data_seed"])).permutation(N)   # one fixed pass, no replacement
    total_steps = math.ceil(N / G)
    quick_val = {l: load_packed(data_dir, "val", l, B, int(ecfg["quick_blocks"])) for l in langs}
    full_val = {l: load_packed(data_dir, "val", l, B, int(ecfg["full_blocks"])) for l in langs}

    # ---- model and optimizer ------------------------------------------------
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                       # identical initialization on every rank
    raw_model = build_model(mcfg, vocab, eos_id, pad_id).to(device)
    torch.manual_seed(seed + rank)                # distinct dropout masks per rank
    if device.type == "cuda":
        torch.cuda.manual_seed(seed + rank)
    n_params = sum(p.numel() for p in raw_model.parameters())
    if tcfg.get("gradient_checkpointing"):
        raw_model.gradient_checkpointing_enable()

    decay = [p for n, p in raw_model.named_parameters() if p.dim() >= 2]      # matrices + embeddings
    no_decay = [p for n, p in raw_model.named_parameters() if p.dim() < 2]    # biases + LayerNorm
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(tcfg["weight_decay"])},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=0.0, betas=tuple(tcfg["betas"]), eps=float(tcfg["eps"]), fused=device.type == "cuda",
    )

    use_bf16 = tcfg["precision"] == "bf16"
    def autocast_ctx():
        if not use_bf16:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    if args.overfit_tiny:
        if world > 1:
            sys.exit("--overfit-tiny runs on one process")
        sys.exit(run_overfit(raw_model, optimizer, train_arrays, cfg, device, autocast_ctx, out_dir, log))

    # ---- Gate 4: learning-rate schedule --------------------------------------
    peak, min_ratio, warmup = float(tcfg["peak_lr"]), float(tcfg["min_lr_ratio"]), int(tcfg["warmup_steps"])
    sched = lambda s: lr_at(s, peak, min_ratio, warmup, total_steps)
    lr_points = {s: sched(s) for s in (1, warmup, total_steps // 2, total_steps)}
    lr_ok = abs(sched(warmup) - peak) < 1e-12 and abs(sched(total_steps) - peak * min_ratio) < 1e-12
    log.info("[gate4] LR schedule " + ("PASS" if lr_ok else "FAIL") + ": " +
             ", ".join(f"step {s}: {v:.3e}" for s, v in lr_points.items()))

    # ---- resume -------------------------------------------------------------
    step = 0
    lang_tokens = np.zeros(L, dtype=np.int64)
    monitor = HealthMonitor(gcfg, langs)
    train_seconds = 0.0
    if args.resume:
        files = checkpoint_files(ckpt_dir)
        path = (files[-1] if files else None) if args.resume == "auto" else Path(args.resume)
        if path is None:
            log.info("[resume] no checkpoint found; starting fresh")
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            raw_model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            step = int(ck["step"])
            lang_tokens = np.array(ck["lang_tokens"], dtype=np.int64)
            monitor.load_state_dict(ck["monitor"])
            train_seconds = float(ck.get("train_seconds", 0.0))
            if ck["world_size"] == world:
                set_rng_state(ck["rng"][rank], device)
            else:
                log.info(f"[resume] WARNING: checkpoint used {ck['world_size']} GPU(s), now {world}; "
                         f"data order is unchanged but dropout RNG streams differ")
            log.info(f"[resume] restored {path} at step {step}")

    model = torch.compile(raw_model) if tcfg.get("compile") else raw_model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None)

    max_steps = min(args.max_steps or total_steps, total_steps)
    accum = math.ceil(G / (mb * world))
    if is_main and step == 0:
        write_json(out_dir / "run_config.json", {
            **cfg, "resolved": {"world_size": world, "grad_accum_per_gpu": accum, "total_steps": total_steps,
                                "total_blocks": N, "total_tokens": N * B, "parameters": n_params,
                                "vocab_size": vocab, "lr_points": lr_points,
                                "torch": torch.__version__}})
    log.info(f"[setup] {n_params:,} parameters | {world} GPU(s) x micro {mb} x accum {accum} = batch {G} | "
             f"{total_steps} steps, {N:,} blocks, {N * B:,} tokens | starting at step {step}")

    # ---- step 0: initial snapshot, initial loss (Gate 4), and generation -------
    ccfg = cfg["checkpoint"]
    tokenizer_dir = Path(cfg["tokenizer_dir"])
    if step == 0 and is_snapshot_step(0, total_steps, ccfg):
        save_snapshot(out_dir, 0, raw_model, tokenizer_dir, lang_tokens, langs, is_main)
        barrier(world)
    if step == 0 and not args.no_eval:
        res = evaluate_blocks(raw_model, quick_val, int(ecfg["micro_batch"]), device, autocast_ctx, rank, world)
        expected = math.log(vocab)
        ok = abs(res["macro_loss"] - expected) <= gcfg["init_loss_tolerance"]
        log.write("eval", {"step": 0, "kind": "quick", **res})
        log.event(0, "PASS" if ok else "FAIL", "gate4", "initial_loss",
                  f"macro loss {res['macro_loss']:.4f}, expected ln({vocab}) = {expected:.4f} "
                  f"+/- {gcfg['init_loss_tolerance']}")
        if is_main and 0 in ecfg["generation_steps"]:
            generate_samples(raw_model, tok, ecfg, eos_id, pad_id, device, autocast_ctx, 0, gcfg, log)
        barrier(world)

    # ---- training loop --------------------------------------------------------
    raw_model.train()
    clip = float(tcfg["grad_clip"])
    log_every = int(cfg["logging"]["interval"])
    win_tokens, win_time = 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < max_steps:
        t0 = time.time()
        ids = order[step * G: min((step + 1) * G, N)]
        targets_global = len(ids) * (B - 1)
        local_ids = ids[rank::world]
        lr = sched(step + 1)
        for group in optimizer.param_groups:
            group["lr"] = lr

        micro = [local_ids[i:i + mb] for i in range(0, len(local_ids), mb)] or [None]
        loss_acc = torch.zeros((), device=device, dtype=torch.float64)
        for j, m in enumerate(micro):
            sync = j == len(micro) - 1
            ctx = model.no_sync() if (world > 1 and not sync) else contextlib.nullcontext()
            with ctx:
                rows = ids[:1] if m is None else m          # empty share: dummy pass to join all-reduce
                x = torch.from_numpy(np.stack([train_arrays[g // bpl][g % bpl] for g in rows]).astype(np.int64))
                x = x.to(device, non_blocking=True)
                with autocast_ctx():
                    loss_sum, _ = lm_loss_sum(model, x)
                scale = 0.0 if m is None else world / targets_global   # DDP averages over ranks
                (loss_sum * scale).backward()
                if m is not None:
                    loss_acc += loss_sum.detach().double()

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if world > 1:
            dist.all_reduce(loss_acc)
        train_loss = loss_acc.item() / targets_global
        step += 1
        lang_tokens += np.bincount(ids // bpl, minlength=L) * B
        dt = time.time() - t0
        train_seconds += dt
        win_tokens += len(ids) * B
        win_time += dt

        if not (math.isfinite(train_loss) and math.isfinite(grad_norm)):
            log.event(step, "STOP", "gate5", "non_finite",
                      f"loss {train_loss} grad_norm {grad_norm}: return to the last checkpoint and investigate")
            sys.exit(EXIT_NAN)
        if monitor.on_step(step, train_loss, grad_norm, clip, log):
            sys.exit(EXIT_SPIKE)

        if step % log_every == 0 or step == max_steps or log_every == 1:
            tps = win_tokens / max(win_time, 1e-9)
            share = lang_tokens / lang_tokens.sum()
            rec = {"step": step, "loss": train_loss, "lr": lr, "grad_norm": grad_norm, "tokens_per_s": tps,
                   "tokens": int(lang_tokens.sum()), "sequences": len(ids),
                   "lang_share": {l: round(float(s), 4) for l, s in zip(langs, share)},
                   "max_mem_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0,
                   "train_hours": train_seconds / 3600}
            log.write("train", rec)
            log.info(f"step {step}/{total_steps} loss {train_loss:.4f} lr {lr:.2e} gnorm {grad_norm:.3f} "
                     f"{tps:,.0f} tok/s mem {rec['max_mem_gb']:.1f}GB")
            monitor.on_throughput(step, tps, log)
            monitor.on_lang_share(step, lang_tokens, log)
            win_tokens, win_time = 0, 0.0

        if not args.no_eval:
            if step % int(ecfg["full_interval"]) == 0 or step == total_steps:
                res = evaluate_blocks(raw_model, full_val, int(ecfg["micro_batch"]), device, autocast_ctx, rank, world)
                log.write("eval", {"step": step, "kind": "full", **res})
                log.info(f"[eval] step {step} full macro {res['macro_loss']:.4f} | " +
                         " ".join(f"{l} {v['loss']:.3f}" for l, v in res["per_language"].items()))
                monitor.on_full_eval(step, res, log)
            elif step % int(ecfg["quick_interval"]) == 0:
                res = evaluate_blocks(raw_model, quick_val, int(ecfg["micro_batch"]), device, autocast_ctx, rank, world)
                log.write("eval", {"step": step, "kind": "quick", **res})
                log.info(f"[eval] step {step} quick macro {res['macro_loss']:.4f}")
            if is_main and (step in ecfg["generation_steps"] or step == total_steps):
                generate_samples(raw_model, tok, ecfg, eos_id, pad_id, device, autocast_ctx, step, gcfg, log)
            barrier(world)

        if is_snapshot_step(step, total_steps, ccfg):
            save_snapshot(out_dir, step, raw_model, tokenizer_dir, lang_tokens, langs, is_main)
            barrier(world)

        if step % int(ccfg["interval"]) == 0 or step == max_steps:
            extra = {"lang_tokens": lang_tokens.tolist(), "monitor": monitor.state_dict(),
                     "rng": gather_rng(world, device), "world_size": world, "config": cfg,
                     "consumed_blocks": min(step * G, N), "train_seconds": train_seconds}
            save_checkpoint(ckpt_dir, step, total_steps, raw_model, optimizer, extra, cfg["checkpoint"], is_main)
            barrier(world)

    # ---- final outputs ---------------------------------------------------------
    if is_main and step == total_steps:
        final_dir = out_dir / "final_model"
        raw_model.save_pretrained(final_dir)
        tok_dir = Path(cfg["tokenizer_dir"])
        for f in tok_dir.iterdir():
            if f.is_file() and f.suffix in (".json", ".txt") and not f.name.startswith("sample_doc_ids"):
                shutil.copy2(f, final_dir / f.name)
        write_json(out_dir / "summary.json", {
            "steps": step,
            "input_tokens_processed": int(lang_tokens.sum()),
            "tokens_per_language": {l: int(t) for l, t in zip(langs, lang_tokens)},
            "equal_exposure": bool(len(set(lang_tokens.tolist())) == 1),
            "parameters": n_params,
            "world_size": world,
            "train_wall_hours": train_seconds / 3600,
            "gpu_hours": train_seconds * world / 3600,
            "mean_tokens_per_s": int(lang_tokens.sum()) / max(train_seconds, 1e-9),
            "max_mem_gb_rank0": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0,
        })
        log.info(f"[done] final model saved to {final_dir}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
