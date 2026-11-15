"""Gate 4 resume test (plan Section 13).

Runs `steps` optimizer steps straight through, then the same run as
steps/2 + save + restart + steps/2, and checks that every per-step loss
matches within the tolerance in configs/train.yaml (gates.resume_tolerance).

  python src/pretrain_gpt2/check_resume.py                 # one GPU
  python src/pretrain_gpt2/check_resume.py --nproc 2       # two GPUs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import iter_jsonl, load_config, write_json


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def losses(out_dir: Path) -> dict[int, float]:
    return {r["step"]: r["loss"] for r in iter_jsonl(out_dir / "train_log.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--base-dir", default="results/debug/resume_test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    base = Path(args.base_dir)
    a_dir, b_dir = base / "continuous", base / "resumed"
    for d in (a_dir, b_dir):
        shutil.rmtree(d, ignore_errors=True)

    launcher = [sys.executable] if args.nproc == 1 else \
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc}"]
    train_script = str(Path(__file__).resolve().with_name("train.py"))
    common = [train_script, "--config", args.config, "--no-eval", "--device", args.device,
              "--set", "logging.interval=1", "--set", "checkpoint.interval=1000000"]
    for o in args.overrides:
        common += ["--set", o]
    half = args.steps // 2

    run(launcher + common + ["--out-dir", str(a_dir), "--max-steps", str(args.steps)])
    run(launcher + common + ["--out-dir", str(b_dir), "--max-steps", str(half)])
    run(launcher + common + ["--out-dir", str(b_dir), "--max-steps", str(args.steps), "--resume", "auto"])

    la, lb = losses(a_dir), losses(b_dir)
    tol = float(cfg["gates"]["resume_tolerance"])
    diffs = {s: abs(la[s] - lb[s]) for s in range(1, args.steps + 1) if s in la and s in lb}
    missing = [s for s in range(1, args.steps + 1) if s not in diffs]
    worst = max(diffs.values()) if diffs else float("inf")
    ok = not missing and worst <= tol
    result = {"steps": args.steps, "resume_at": half, "nproc": args.nproc, "tolerance": tol,
              "max_abs_loss_diff": worst, "missing_steps": missing,
              "per_step_diff": {str(s): d for s, d in diffs.items()}, "pass": ok}
    write_json(base / "gate4_resume.json", result)
    print(f"[gate4] resume {'PASS' if ok else 'FAIL'}: max |loss diff| {worst:.2e} (tolerance {tol:.0e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
