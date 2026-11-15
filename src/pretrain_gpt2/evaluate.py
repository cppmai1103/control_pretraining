"""Per-language held-out evaluation and fixed-prompt generation (plan Section 12).

  python src/pretrain_gpt2/evaluate.py --model results/gpt2_5lang_500M/final_model --split val
  python src/pretrain_gpt2/evaluate.py --model results/gpt2_5lang_500M/checkpoints/step_003815.pt --split val
  python src/pretrain_gpt2/evaluate.py --model results/gpt2_5lang_500M/final_model --split test --confirm-test

The test split is for final evaluation after model selection only; it needs
--confirm-test so it is not used by accident to choose settings.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

import torch

from common import add_override_arg, build_model, evaluate_blocks, load_config, load_packed, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="final_model directory or a step_*.pt checkpoint")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--blocks", type=int, default=None, help="blocks per language (default: all held-out)")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--generate", action="store_true", help="also write fixed-prompt generations")
    parser.add_argument("--out", default=None, help="output JSON (default: next to the model)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    add_override_arg(parser)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        sys.exit("Refusing to evaluate on test without --confirm-test (use it only after model selection).")

    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data_dir = Path(cfg["data_dir"])
    meta = read_json(data_dir / "meta.json")
    langs, B = list(cfg["languages"]), int(cfg["train"]["block_size"])
    n_blocks = args.blocks or int(meta["heldout_blocks_per_language"])

    model_path = Path(args.model)
    if model_path.suffix == ".pt":
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        model = build_model(ck["config"]["model"], meta["vocab_size"], meta["eos_id"], meta["pad_id"])
        model.load_state_dict(ck["model"])
        step = ck["step"]
    else:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_path)
        step = None
    model.to(device)

    use_bf16 = cfg["train"]["precision"] == "bf16"
    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    arrays = {l: load_packed(data_dir, args.split, l, B, n_blocks) for l in langs}
    res = evaluate_blocks(model, arrays, int(cfg["eval"]["micro_batch"]), device, autocast_ctx)
    res.update({"model": str(model_path), "step": step, "split": args.split, "blocks_per_language": n_blocks,
                "macro_ppl": math.exp(res["macro_loss"])})

    if args.generate:
        from tokenizers import Tokenizer
        from train import RunLog, generate_samples
        tok = Tokenizer.from_file(str(Path(cfg["tokenizer_dir"]) / "tokenizer.json"))
        tok.encode_special_tokens = True
        out_dir = model_path.parent if model_path.suffix == ".pt" else model_path
        log = RunLog(out_dir, True)
        log.gen_path = out_dir / f"generations_eval_{model_path.stem}.jsonl"
        log.event_path = out_dir / f"events_eval_{model_path.stem}.jsonl"
        generate_samples(model, tok, cfg["eval"], meta["eos_id"], meta["pad_id"], device, autocast_ctx,
                         step or 0, cfg["gates"], log)

    out = Path(args.out) if args.out else (model_path.parent if model_path.suffix == ".pt" else model_path) / \
        f"eval_{args.split}_{model_path.stem}.json"
    write_json(out, res)
    print(f"[evaluate] {args.split} macro loss {res['macro_loss']:.4f} (ppl {res['macro_ppl']:.2f})")
    for l, v in res["per_language"].items():
        print(f"  {l}: loss {v['loss']:.4f}  ppl {v['ppl']:.2f}")
    print(f"[evaluate] written to {out}")


if __name__ == "__main__":
    main()
