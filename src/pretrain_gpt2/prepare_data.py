"""Collect, clean, deduplicate, and split the five-language corpus (plan Sections 3-5, Gate 1).

Stages:
  collect  stream a seeded sample of each source into data/raw/<lang>.jsonl
  clean    normalize, filter, deduplicate, assign splits -> data/clean/<lang>/{train,val,test}.jsonl
  all      both

Examples:
  python src/pretrain_gpt2/prepare_data.py --stage all
  python src/pretrain_gpt2/prepare_data.py --stage clean --langs ja zh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from common import (add_override_arg, append_jsonl, clean_path, ensure_dir, iter_jsonl, load_config,
                    split_for_group, stage_rng, write_json)

URL_RE = re.compile(r"(?:https?://|www\.)\S+")
CHAR_RUN_RE = re.compile(r"(.)\1{9,}", re.DOTALL)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NEAR_DUP_STRIP_RE = re.compile(r"[\W_]+", re.UNICODE)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def resolve_revision(dataset: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset, revision=revision).sha


def source_files(dataset: str, config: str, sha: str) -> list[str]:
    from datasets import load_dataset_builder

    builder = load_dataset_builder(dataset, name=config, revision=sha)
    files = builder.config.data_files["train"]
    return sorted(str(f) for f in files)


def collect(cfg: dict, lang: str) -> None:
    from datasets import interleave_datasets, load_dataset

    spec = cfg["languages"][lang]
    seed = cfg["seed"]
    raw_dir = ensure_dir(Path(cfg["data_root"]) / "raw")
    out_path = raw_dir / f"{lang}.jsonl"
    meta_path = raw_dir / f"{lang}.meta.json"
    if meta_path.exists():
        print(f"[collect] {lang}: already complete ({meta_path}); delete it to recollect")
        return

    sha = resolve_revision(spec["dataset"], spec["revision"])
    files = source_files(spec["dataset"], spec["config"], sha)
    rng = stage_rng(seed)
    n_files = min(len(files), int(cfg["collection"].get("files_per_language", 16)))
    chosen = sorted(rng.sample(files, n_files))
    print(f"[collect] {lang}: {spec['dataset']}/{spec['config']}@{sha[:10]}, "
          f"{n_files} of {len(files)} source files")

    # One stream per chosen file, each with its own shuffle buffer, interleaved
    # round-robin so the sample spreads across files and crawl snapshots.
    per_file_buffer = max(1, cfg["collection"]["shuffle_buffer"] // n_files)
    streams = [
        load_dataset("parquet", data_files={"train": [f]}, split="train", streaming=True)
        .shuffle(seed=seed, buffer_size=per_file_buffer)
        for f in chosen
    ]
    ds = interleave_datasets(streams, seed=seed, stopping_strategy="all_exhausted")

    target = int(float(spec["target_bytes"]))
    total_bytes = 0
    n_docs = 0
    per_dump = Counter()
    tmp_path = out_path.with_suffix(".jsonl.partial")
    t0 = time.time()
    with open(tmp_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex.get("text") or ""
            record = {
                "doc_id": ex.get("id"),
                "language": lang,
                "text": text,
                "url": ex.get("url"),
                "source_file": ex.get("file_path"),
                "dump": ex.get("dump"),
                "language_score": ex.get("language_score"),
                "dataset": spec["dataset"],
                "dataset_config": spec["config"],
                "dataset_revision": sha,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_bytes += len(text.encode("utf-8"))
            n_docs += 1
            per_dump[record["dump"]] += 1
            if n_docs % 50000 == 0:
                rate = total_bytes / max(1e-9, time.time() - t0) / 1e6
                print(f"[collect] {lang}: {n_docs:,} docs, {total_bytes / 1e9:.2f}/{target / 1e9:.2f} GB "
                      f"({rate:.1f} MB/s)", flush=True)
            if total_bytes >= target:
                break
    tmp_path.rename(out_path)
    write_json(meta_path, {
        "language": lang,
        "dataset": spec["dataset"],
        "dataset_config": spec["config"],
        "requested_revision": spec["revision"],
        "dataset_revision": sha,
        "chosen_source_files": chosen,
        "available_source_files": len(files),
        "seed": seed,
        "shuffle_buffer_total": cfg["collection"]["shuffle_buffer"],
        "shuffle_buffer_per_file": per_file_buffer,
        "documents": n_docs,
        "utf8_bytes": total_bytes,
        "target_bytes": target,
        "documents_per_dump": dict(per_dump),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"[collect] {lang}: done, {n_docs:,} docs, {total_bytes / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    return unicodedata.normalize("NFC", text).strip()


def reject_reason(text: str, score, ccfg: dict) -> str | None:
    n = len(text)
    if n == 0:
        return "empty"
    if n < ccfg["min_chars"]:
        return "too_short"
    if text.count("�") / n > ccfg["max_replacement_char_fraction"]:
        return "invalid_text"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 3 and 1 - len(set(lines)) / len(lines) > ccfg["max_duplicate_line_fraction"]:
        return "duplicate_lines"
    if sum(len(m.group(0)) for m in CHAR_RUN_RE.finditer(text)) / n > ccfg["max_char_run_fraction"]:
        return "repeated_chars"
    if sum(len(m.group(0)) for m in URL_RE.finditer(text)) / n > ccfg["max_url_char_fraction"]:
        return "url_heavy"
    threshold = ccfg.get("min_language_score")
    if threshold is not None and score is not None and score < threshold:
        return "low_language_score"
    return None


def near_dup_key(text: str) -> str:
    """Group key: case-folded NFKC text without punctuation/whitespace.

    Documents differing only in case, spacing, or punctuation share a group,
    so they always land in the same split. (FineWeb/FineWeb2 are already
    MinHash-deduplicated upstream; this catches the remaining trivial variants.)
    """
    key = NEAR_DUP_STRIP_RE.sub("", unicodedata.normalize("NFKC", text).casefold())
    return hashlib.blake2b(key.encode("utf-8"), digest_size=12).hexdigest()


class Reservoir:
    def __init__(self, k: int, rng):
        self.k, self.rng, self.items, self.seen = k, rng, [], 0

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.items[j] = item


def clean(cfg: dict, lang: str) -> dict:
    ccfg = cfg["cleaning"]
    scfg = cfg["split"]
    seed = cfg["seed"]
    root = Path(cfg["data_root"])
    raw_path = root / "raw" / f"{lang}.jsonl"
    if not raw_path.exists():
        sys.exit(f"[clean] missing {raw_path}; run --stage collect first")

    out_dir = ensure_dir(root / "clean" / lang)
    outs = {s: open(out_dir / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    inspect_dir = ensure_dir(root / "reports" / "inspect")
    kept_sample = Reservoir(ccfg["inspect_kept"], stage_rng(seed))
    rejected_sample = Reservoir(ccfg["inspect_rejected"], stage_rng(seed))

    reasons = Counter()
    seen_exact: set[bytes] = set()
    group_sizes = Counter()
    group_split: dict[str, str] = {}
    doc_ids_by_split = {s: set() for s in outs}
    split_docs = Counter()
    split_bytes = Counter()
    collected = 0
    collected_bytes = 0
    scores = []

    for rec in iter_jsonl(raw_path):
        collected += 1
        raw_text = rec.get("text") or ""
        collected_bytes += len(raw_text.encode("utf-8"))
        text = normalize(raw_text)
        reason = reject_reason(text, rec.get("language_score"), ccfg)
        if reason is None:
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            if h in seen_exact:
                reason = "exact_duplicate"
            else:
                seen_exact.add(h)
        if reason is not None:
            reasons[reason] += 1
            rejected_sample.add({"reason": reason, "doc_id": rec.get("doc_id"), "text": text[:3000]})
            continue

        group = near_dup_key(text)
        split = split_for_group(group, seed, scfg["train_frac"], scfg["val_frac"])
        group_sizes[group] += 1
        group_split.setdefault(group, split)
        if rec.get("language_score") is not None:
            scores.append(rec["language_score"])
        out = {
            "doc_id": rec.get("doc_id"),
            "language": lang,
            "text": text,
            "url": rec.get("url"),
            "source_file": rec.get("source_file"),
            "dump": rec.get("dump"),
            "dataset_revision": rec.get("dataset_revision"),
            "group_id": group,
        }
        outs[split].write(json.dumps(out, ensure_ascii=False) + "\n")
        doc_ids_by_split[split].add(rec.get("doc_id"))
        split_docs[split] += 1
        split_bytes[split] += len(text.encode("utf-8"))
        kept_sample.add({"doc_id": rec.get("doc_id"), "url": rec.get("url"), "text": text[:3000]})

    for f in outs.values():
        f.close()

    for name, sample in (("kept", kept_sample), ("rejected", rejected_sample)):
        path = inspect_dir / f"{lang}_{name}.jsonl"
        path.unlink(missing_ok=True)
        for item in sample.items:
            append_jsonl(path, item)

    # Gate 1 leakage checks: groups map to one split by construction; verify anyway.
    group_conflicts = sum(1 for g in group_sizes if split_for_group(g, seed, scfg["train_frac"],
                                                                     scfg["val_frac"]) != group_split[g])
    id_overlap = {
        "train_val": len(doc_ids_by_split["train"] & doc_ids_by_split["val"]),
        "train_test": len(doc_ids_by_split["train"] & doc_ids_by_split["test"]),
        "val_test": len(doc_ids_by_split["val"] & doc_ids_by_split["test"]),
    }
    kept = sum(split_docs.values())
    scores.sort()
    summary = {
        "language": lang,
        "collected_documents": collected,
        "collected_utf8_bytes": collected_bytes,
        "kept_documents": kept,
        "kept_utf8_bytes": sum(split_bytes.values()),
        "rejections": dict(reasons),
        "exact_duplicate_rate": reasons["exact_duplicate"] / max(1, collected),
        "near_duplicate_groups_with_multiple_docs": sum(1 for v in group_sizes.values() if v > 1),
        "documents_in_multi_doc_groups": sum(v for v in group_sizes.values() if v > 1),
        "split_documents": dict(split_docs),
        "split_utf8_bytes": dict(split_bytes),
        "language_score": None if not scores else {
            "p05": scores[int(0.05 * (len(scores) - 1))],
            "median": scores[len(scores) // 2],
            "min": scores[0],
        },
        "cleaning_thresholds": ccfg,
        "gate1": {
            "doc_id_overlap_between_splits": id_overlap,
            "group_split_conflicts": group_conflicts,
            "split_leakage_pass": group_conflicts == 0 and not any(id_overlap.values()),
            "manual_inspection_files": [str(inspect_dir / f"{lang}_kept.jsonl"),
                                        str(inspect_dir / f"{lang}_rejected.jsonl")],
            "language_purity": "PENDING manual review: record the share of the kept sample that is "
                               "predominantly in the target language (pass >= 0.95)",
        },
    }
    write_json(root / "reports" / f"cleaning_{lang}.json", summary)
    status = "PASS" if summary["gate1"]["split_leakage_pass"] else "FAIL"
    print(f"[clean] {lang}: kept {kept:,}/{collected:,} docs, {summary['kept_utf8_bytes'] / 1e9:.2f} GB; "
          f"splits {dict(split_docs)}; Gate 1 leakage {status}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--stage", choices=["collect", "clean", "all"], default="all")
    parser.add_argument("--langs", nargs="+", default=None)
    add_override_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    langs = args.langs or list(cfg["languages"])

    if args.stage in ("collect", "all"):
        for lang in langs:
            collect(cfg, lang)
    if args.stage in ("clean", "all"):
        summaries = {lang: clean(cfg, lang) for lang in langs}
        failed = [l for l, s in summaries.items() if not s["gate1"]["split_leakage_pass"]]
        if failed:
            sys.exit(f"[clean] Gate 1 split-leakage check FAILED for {failed}")
        print("[clean] Gate 1 automatic checks passed. Next: review data/reports/inspect/* by hand "
              "(language purity >= 95%), then train the tokenizer.")


if __name__ == "__main__":
    main()
