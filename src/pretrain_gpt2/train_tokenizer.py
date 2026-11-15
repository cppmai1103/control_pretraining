"""Train the shared byte-level BPE tokenizer (plan Section 6).

Samples ~100MB of UTF-8 text per language from the *training* split only
(seed 42), then trains one 64,000-entry vocabulary for all five languages
with GPT-2 pre-tokenization (regex split, add_prefix_space=False).

Run inspect_tokenizer.py afterwards; the tokenizer is frozen only after Gate 2 passes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (add_override_arg, clean_path, ensure_dir, line_offsets, load_config, read_jsonl_at,
                    stage_rng, write_json)


def build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def sample_language(cfg: dict, lang: str) -> tuple[list[int], int, list[str]]:
    """Choose training documents (seed 42) until ~sample_bytes_per_language UTF-8 bytes."""
    path = clean_path(cfg["data_root"], lang, "train")
    offsets = line_offsets(path)
    order = list(range(len(offsets)))
    stage_rng(cfg["seed"]).shuffle(order)
    budget = int(float(cfg["tokenizer"]["sample_bytes_per_language"]))
    chosen, total, ids = [], 0, []
    with open(path, encoding="utf-8") as f:
        for i in order:
            rec = read_jsonl_at(f, offsets[i])
            chosen.append(offsets[i])
            ids.append(rec.get("doc_id"))
            total += len(rec["text"].encode("utf-8"))
            if total >= budget:
                break
    if total < budget:
        print(f"[tokenizer] WARNING {lang}: only {total / 1e6:.1f}MB of training text available "
              f"(target {budget / 1e6:.0f}MB)")
    return sorted(chosen), total, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    add_override_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    from tokenizers import pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tcfg = cfg["tokenizer"]
    langs = list(cfg["languages"])
    out_dir = ensure_dir(Path(cfg["data_root"]) / tcfg["out_dir"])

    samples = {}
    manifest = {}
    for lang in langs:
        offsets, nbytes, ids = sample_language(cfg, lang)
        samples[lang] = offsets
        manifest[lang] = {"documents": len(offsets), "utf8_bytes": nbytes}
        with open(out_dir / f"sample_doc_ids_{lang}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")
        print(f"[tokenizer] {lang}: {len(offsets):,} docs, {nbytes / 1e6:.1f}MB sampled")

    def text_iterator():
        for lang in langs:
            with open(clean_path(cfg["data_root"], lang, "train"), encoding="utf-8") as f:
                for off in samples[lang]:
                    yield read_jsonl_at(f, off)["text"]

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=int(tcfg["vocab_size"]),
        min_frequency=int(tcfg["min_frequency"]),
        special_tokens=list(tcfg["special_tokens"]),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.time()
    tok.train_from_iterator(text_iterator(), trainer=trainer,
                            length=sum(len(v) for v in samples.values()))
    print(f"[tokenizer] trained in {time.time() - t0:.0f}s, vocabulary {tok.get_vocab_size():,}")

    tok.save(str(out_dir / "tokenizer.json"))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<eos>",
        bos_token="<eos>",   # GPT-2 convention: BOS = EOS
        pad_token="<pad>",
        model_max_length=1024,
    )
    hf_tok.save_pretrained(str(out_dir))

    write_json(out_dir / "tokenizer_settings.json", {
        "algorithm": "byte-level BPE",
        "vocab_size_requested": int(tcfg["vocab_size"]),
        "vocab_size_actual": tok.get_vocab_size(),
        "normalizer": "NFC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False, use_regex=True)  # GPT-2 split regex",
        "number_splitting": "none (GPT-2 default)",
        "decoder": "ByteLevel",
        "initial_alphabet": "all 256 bytes",
        "special_tokens": {t: tok.token_to_id(t) for t in tcfg["special_tokens"]},
        "bos_token": "<eos>",
        "min_frequency": int(tcfg["min_frequency"]),
        "seed": cfg["seed"],
        "sample": manifest,
        "seconds": round(time.time() - t0, 1),
    })
    print(f"[tokenizer] saved to {out_dir}. Next: python src/pretrain_gpt2/inspect_tokenizer.py (Gate 2)")


if __name__ == "__main__":
    main()
