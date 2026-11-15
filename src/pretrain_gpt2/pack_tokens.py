"""Tokenize and pack monolingual 1,024-token blocks (plan Section 7, Gates 1 and 3).

For each language: shuffle documents (seed 42), encode without automatic
special tokens, append one <eos> per document, concatenate, and cut
1,024-token blocks. Training stops at exactly 97,656 blocks per language;
validation and test get 1,024 blocks each.

Outputs (under data_root/packed):
  {train,val,test}_<lang>.bin  uint16 token IDs, shape (blocks, 1024)
  manifest_<split>_<lang>.txt  document IDs consumed, in order
  meta.json                    counts, tokenizer IDs, Gate 3 results
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from common import (add_override_arg, clean_path, ensure_dir, line_offsets, load_config, packed_path,
                    read_jsonl_at, stage_rng, write_json)


def pack_split(cfg: dict, tok, lang: str, split: str, n_blocks: int, out_dir: Path, eos_id: int) -> dict:
    pcfg = cfg["packing"]
    block = int(pcfg["block_size"])
    src = clean_path(cfg["data_root"], lang, split)
    offsets = line_offsets(src)
    order = list(range(len(offsets)))
    stage_rng(cfg["seed"]).shuffle(order)

    out_path = packed_path(out_dir, split, lang)
    arr = np.memmap(Path(str(out_path) + ".partial"), dtype=np.uint16, mode="w+", shape=(n_blocks, block))
    buf: list[int] = []
    filled = 0
    docs_used = 0
    tokens_seen = 0
    batch_size = int(pcfg["encode_batch_docs"])
    manifest = open(out_dir / f"manifest_{split}_{lang}.txt", "w", encoding="utf-8")

    with open(src, encoding="utf-8") as f:
        for start in range(0, len(order), batch_size):
            recs = [read_jsonl_at(f, offsets[i]) for i in order[start:start + batch_size]]
            encs = tok.encode_batch([r["text"] for r in recs])
            for rec, enc in zip(recs, encs):
                ids = enc.ids
                ids.append(eos_id)
                tokens_seen += len(ids)
                buf.extend(ids)
                docs_used += 1
                manifest.write(f"{rec.get('doc_id')}\n")
                while len(buf) >= block and filled < n_blocks:
                    arr[filled] = buf[:block]
                    del buf[:block]
                    filled += 1
                if filled == n_blocks:
                    break
            if filled == n_blocks:
                break
    manifest.close()
    arr.flush()
    del arr
    if filled < n_blocks:
        Path(str(out_path) + ".partial").unlink(missing_ok=True)
        return {"ok": False, "blocks": filled, "needed": n_blocks, "documents": docs_used,
                "available_tokens": tokens_seen, "documents_available": len(offsets)}
    Path(str(out_path) + ".partial").replace(out_path)
    return {"ok": True, "blocks": filled, "documents": docs_used, "documents_available": len(offsets),
            "tokens": filled * block, "dropped_remainder_tokens": len(buf)}


def block_hashes(arr: np.ndarray) -> list[bytes]:
    return [hashlib.blake2b(row.tobytes(), digest_size=8).digest() for row in arr]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    add_override_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    from tokenizers import Tokenizer

    pcfg = cfg["packing"]
    block = int(pcfg["block_size"])
    langs = list(cfg["languages"])
    root = Path(cfg["data_root"])
    tok = Tokenizer.from_file(str(root / cfg["tokenizer"]["out_dir"] / "tokenizer.json"))
    tok.encode_special_tokens = True
    tok.no_padding()
    tok.no_truncation()
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")
    pad_id = tok.token_to_id("<pad>")
    if vocab > 65536:
        sys.exit(f"vocabulary {vocab} does not fit in uint16")
    out_dir = ensure_dir(root / "packed")

    targets = {"train": int(pcfg["train_blocks_per_language"]),
               "val": int(pcfg["heldout_blocks_per_language"]),
               "test": int(pcfg["heldout_blocks_per_language"])}
    results: dict = {}
    shortfalls = []
    for lang in langs:
        results[lang] = {}
        for split, n in targets.items():
            r = pack_split(cfg, tok, lang, split, n, out_dir, eos_id)
            results[lang][split] = r
            status = "ok" if r["ok"] else f"SHORT ({r['blocks']}/{n} blocks)"
            print(f"[pack] {lang}/{split}: {status}, {r['documents']:,} docs used", flush=True)
            if not r["ok"]:
                shortfalls.append(f"{lang}/{split}")

    if shortfalls:
        write_json(out_dir / "meta.json", {"complete": False, "results": results})
        sys.exit(f"[pack] Gate 1 FAIL: not enough text for {shortfalls}. Collect more source text "
                 f"(raise target_bytes in configs/data.yaml), rerun prepare_data.py, then pack again.")

    # Gate 3 -------------------------------------------------------------------
    rng = stage_rng(cfg["seed"])
    train_hashes: dict[bytes, str] = {}
    duplicates = 0
    max_id = 0
    samples = []
    for lang in langs:
        arr = np.memmap(packed_path(out_dir, "train", lang), dtype=np.uint16, mode="r").reshape(-1, block)
        assert arr.shape[0] == targets["train"]
        max_id = max(max_id, int(arr.max()))
        for h in block_hashes(arr):
            if h in train_hashes:
                duplicates += 1
            train_hashes[h] = lang
        for i in sorted(rng.sample(range(arr.shape[0]), 5)):
            samples.append((lang, i, tok.decode(arr[i].tolist(), skip_special_tokens=False)))
    heldout_overlap = 0
    for lang in langs:
        for split in ("val", "test"):
            arr = np.memmap(packed_path(out_dir, split, lang), dtype=np.uint16, mode="r").reshape(-1, block)
            max_id = max(max_id, int(arr.max()))
            heldout_overlap += sum(1 for h in block_hashes(arr) if h in train_hashes)

    sample_path = root / "reports" / "gate3_decoded_blocks.txt"
    ensure_dir(sample_path.parent)
    with open(sample_path, "w", encoding="utf-8") as f:
        for lang, i, text in samples:
            f.write(f"===== {lang} train block {i} =====\n{text}\n\n")

    gate3 = {
        "block_counts": {l: results[l]["train"]["blocks"] for l in langs},
        "block_counts_pass": all(results[l]["train"]["blocks"] == targets["train"] for l in langs),
        "max_token_id": max_id,
        "vocab_size": vocab,
        "token_range_pass": max_id < vocab,
        "duplicate_train_blocks": duplicates,
        "train_heldout_identical_blocks": heldout_overlap,
        "duplicates_pass": duplicates == 0 and heldout_overlap == 0,
        "readability": f"MANUAL: read {sample_path} (5 decoded blocks per language)",
    }
    gate3["pass"] = gate3["block_counts_pass"] and gate3["token_range_pass"] and gate3["duplicates_pass"]
    write_json(out_dir / "meta.json", {
        "complete": True,
        "block_size": block,
        "dtype": "uint16",
        "vocab_size": vocab,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "languages": langs,
        "train_blocks_per_language": targets["train"],
        "heldout_blocks_per_language": targets["val"],
        "quick_val_blocks": int(pcfg["quick_val_blocks"]),
        "total_train_tokens": targets["train"] * block * len(langs),
        "seed": cfg["seed"],
        "results": results,
        "gate3": gate3,
    })
    print(f"[pack] Gate 3 {'PASS' if gate3['pass'] else 'FAIL'}: {gate3}")
    if not gate3["pass"]:
        sys.exit(1)
    print(f"[pack] Also read {sample_path} by hand before training.")


if __name__ == "__main__":
    main()
