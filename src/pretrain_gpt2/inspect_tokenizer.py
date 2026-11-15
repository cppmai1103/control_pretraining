"""Gate 2: check the shared tokenizer before freezing it (plan Sections 6 and 13).

Writes data/reports/gate2_tokenizer.{json,md} and exits non-zero if any
automatic check fails. Checks:
  1. exact round trips on held-out (validation) documents
  2. vocabulary size and special-token safety
  3. cross-language parity on FLORES dev: tokens(lang) / tokens(en)
  4. ja/zh tokens that are not whole UTF-8 characters
  5. comparison against a reference multilingual tokenizer
plus fertility, document-length statistics, and readable example tokenizations.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

from common import (add_override_arg, clean_path, ensure_dir, line_offsets, load_config, read_jsonl_at,
                    stage_rng, write_json)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-character map used by ByteLevel."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_is_whole_chars(token: str, char_to_byte: dict[str, int]) -> bool:
    try:
        bytes(char_to_byte[c] for c in token).decode("utf-8")
        return True
    except (KeyError, UnicodeDecodeError):
        return False


def heldout_sample(cfg: dict, lang: str, n: int) -> list[str]:
    path = clean_path(cfg["data_root"], lang, "val")
    offsets = line_offsets(path)
    order = list(range(len(offsets)))
    stage_rng(cfg["seed"]).shuffle(order)
    with open(path, encoding="utf-8") as f:
        return [read_jsonl_at(f, offsets[i])["text"] for i in order[:n]]


def percentiles(values: list[int]) -> dict:
    v = sorted(values)
    if not v:
        return {}
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1)))]
    return {"p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99), "max": v[-1]}


def load_flores(icfg: dict, langs: list[str]) -> dict[str, list[str]]:
    from datasets import load_dataset

    fl = icfg["flores"]
    out = {}
    for lang in langs:
        ds = load_dataset(fl["dataset"], fl["configs"][lang], split=fl["split"])
        if "id" in ds.column_names:
            ds = ds.sort("id")
        out[lang] = list(ds["text"])
    lengths = {len(v) for v in out.values()}
    if len(lengths) != 1:
        raise ValueError(f"FLORES splits are not aligned: {[(l, len(v)) for l, v in out.items()]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    add_override_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    from tokenizers import Tokenizer

    langs = list(cfg["languages"])
    icfg = cfg["inspection"]
    gcfg = cfg["gates"]["tokenizer"]
    tok_dir = Path(cfg["data_root"]) / cfg["tokenizer"]["out_dir"]
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    tok.encode_special_tokens = True  # literal "<eos>" in text must stay ordinary text
    report_dir = ensure_dir(Path(cfg["data_root"]) / "reports")

    checks: dict[str, dict] = {}
    report: dict = {"checks": checks}

    # 2. Vocabulary and special tokens ---------------------------------------
    special = {t: tok.token_to_id(t) for t in cfg["tokenizer"]["special_tokens"]}
    literal = "before <eos> middle <pad> after"
    literal_ids = tok.encode(literal).ids
    leaked = [t for t, i in special.items() if i in literal_ids]
    vocab = tok.get_vocab_size()
    checks["vocab_and_special_tokens"] = {
        "vocab_size": vocab,
        "expected": int(cfg["tokenizer"]["vocab_size"]),
        "special_token_ids": special,
        "literal_strings_became_control_tokens": leaked,
        "pass": vocab == int(cfg["tokenizer"]["vocab_size"]) and None not in special.values() and not leaked,
    }

    # 1. Round trip, fertility, lengths on held-out documents -----------------
    b2u = bytes_to_unicode()
    u2b = {c: b for b, c in b2u.items()}
    special_ids = set(special.values())
    whole = [token_is_whole_chars(tok.id_to_token(i), u2b) if i not in special_ids else True
             for i in range(vocab)]

    stats = {}
    round_trip_failures = []
    for lang in langs:
        docs = heldout_sample(cfg, lang, int(icfg["heldout_docs_per_language"]))
        encs = tok.encode_batch(docs)
        n_tok = n_chars = n_bytes = partial = 0
        lengths = []
        fails = 0
        for text, enc in zip(docs, encs):
            ids = enc.ids
            n_tok += len(ids)
            n_chars += len(text)
            n_bytes += len(text.encode("utf-8"))
            lengths.append(len(ids))
            partial += sum(1 for i in ids if not whole[i])
            if tok.decode(ids, skip_special_tokens=False) != text:
                fails += 1
                if len(round_trip_failures) < 5:
                    round_trip_failures.append({"language": lang, "text": text[:300]})
        stats[lang] = {
            "documents": len(docs),
            "round_trip_failures": fails,
            "tokens_per_char": n_tok / max(1, n_chars),
            "tokens_per_byte": n_tok / max(1, n_bytes),
            "chars_per_token": n_chars / max(1, n_tok),
            "partial_char_token_fraction": partial / max(1, n_tok),
            "doc_length_tokens": percentiles(lengths),
        }
        print(f"[gate2] {lang}: tokens/char {stats[lang]['tokens_per_char']:.3f}, "
              f"partial {stats[lang]['partial_char_token_fraction']:.4f}, round-trip failures {fails}")
    report["heldout_stats"] = stats
    checks["round_trip"] = {
        "failures": {l: s["round_trip_failures"] for l, s in stats.items()},
        "examples": round_trip_failures,
        "pass": all(s["round_trip_failures"] == 0 and s["documents"] > 0 for s in stats.values()),
    }

    # 4. Partial UTF-8 characters in ja/zh ------------------------------------
    cjk = [l for l in ("ja", "zh") if l in stats]
    checks["partial_byte_tokens"] = {
        "fraction": {l: stats[l]["partial_char_token_fraction"] for l in cjk},
        "threshold": gcfg["max_partial_byte_fraction"],
        "pass": all(stats[l]["partial_char_token_fraction"] < gcfg["max_partial_byte_fraction"] for l in cjk),
    }

    # 3. FLORES parity ---------------------------------------------------------
    flores = None
    try:
        flores = load_flores(icfg, langs)
        counts = {l: sum(len(e.ids) for e in tok.encode_batch(flores[l])) for l in langs}
        parity = {l: counts[l] / counts["en"] for l in langs}
        verdict = {l: ("PASS" if p <= gcfg["max_parity_pass"] else
                       "WARN" if p <= gcfg["max_parity_fail"] else "FAIL") for l, p in parity.items()}
        checks["flores_parity"] = {
            "sentences": len(flores["en"]),
            "tokens": counts,
            "parity_vs_en": parity,
            "verdict": verdict,
            "pass_threshold": gcfg["max_parity_pass"],
            "fail_threshold": gcfg["max_parity_fail"],
            "pass": all(v == "PASS" for v in verdict.values()),
        }
        print(f"[gate2] FLORES parity vs en: " + ", ".join(f"{l} {p:.2f}" for l, p in parity.items()))
    except Exception as exc:  # gated dataset, no network, renamed config...
        checks["flores_parity"] = {
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "fix": "Accept the FLORES+ terms on the Hugging Face Hub, export HF_TOKEN, and check "
                   "inspection.flores.configs in configs/data.yaml.",
        }
        print(f"[gate2] FLORES parity could not run: {exc}")

    # 5. Reference tokenizer ---------------------------------------------------
    ref_name = icfg.get("reference_tokenizer")
    if ref_name:
        try:
            from transformers import AutoTokenizer

            ref = AutoTokenizer.from_pretrained(ref_name)
            ratio = {}
            ref_tpc = {}
            for lang in langs:
                docs = heldout_sample(cfg, lang, min(500, int(icfg["heldout_docs_per_language"])))
                n_ref = sum(len(x) for x in ref(docs, add_special_tokens=False)["input_ids"])
                n_chars = sum(len(d) for d in docs)
                ref_tpc[lang] = n_ref / max(1, n_chars)
                ratio[lang] = stats[lang]["tokens_per_char"] / max(1e-9, ref_tpc[lang])
            med = statistics.median(ratio.values())
            flagged = [l for l, r in ratio.items() if r > gcfg["max_reference_ratio_vs_median"] * med]
            checks["reference_comparison"] = {
                "reference": ref_name,
                "reference_tokens_per_char": ref_tpc,
                "ours_over_reference": ratio,
                "median_ratio": med,
                "disproportionately_worse": flagged,
                "pass": not flagged,
            }
            if flores is not None:
                rc = {l: sum(len(x) for x in ref(flores[l], add_special_tokens=False)["input_ids"]) for l in langs}
                checks["reference_comparison"]["reference_flores_parity_vs_en"] = {l: rc[l] / rc["en"] for l in langs}
            print(f"[gate2] ours/reference tokens-per-char: " + ", ".join(f"{l} {r:.2f}" for l, r in ratio.items()))
        except Exception as exc:
            checks["reference_comparison"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[gate2] reference comparison could not run: {exc}")

    # Readable example tokenizations ------------------------------------------
    examples = {}
    for lang, text in icfg.get("examples", {}).items():
        ids = tok.encode(text).ids
        examples[lang] = {"text": text, "n_tokens": len(ids),
                          "pieces": [tok.decode([i], skip_special_tokens=False) for i in ids]}
    report["examples"] = examples

    passed = all(c.get("pass") for c in checks.values())
    report["gate2_pass"] = passed
    write_json(report_dir / "gate2_tokenizer.json", report)

    lines = ["# Gate 2: tokenizer report", "", f"**Overall: {'PASS' if passed else 'FAIL'}**", "",
             "| Check | Result |", "|---|---|"]
    for name, c in checks.items():
        lines.append(f"| {name} | {'PASS' if c.get('pass') else 'FAIL'} |")
    lines += ["", "## Held-out statistics", "",
              "| Language | Tokens/char | Tokens/byte | Partial-char tokens | Median doc tokens |",
              "|---|---:|---:|---:|---:|"]
    for l, s in stats.items():
        lines.append(f"| {l} | {s['tokens_per_char']:.3f} | {s['tokens_per_byte']:.3f} | "
                     f"{s['partial_char_token_fraction']:.4f} | {s['doc_length_tokens'].get('p50', 0)} |")
    if "parity_vs_en" in checks.get("flores_parity", {}):
        lines += ["", "## FLORES dev parity (tokens / English tokens)", "", "| Language | Parity | Verdict |",
                  "|---|---:|---|"]
        for l, p in checks["flores_parity"]["parity_vs_en"].items():
            lines.append(f"| {l} | {p:.2f} | {checks['flores_parity']['verdict'][l]} |")
    lines += ["", "## Example tokenizations", ""]
    for l, e in examples.items():
        lines += [f"**{l}** ({e['n_tokens']} tokens): {e['text']}", "",
                  "`" + " | ".join(p.replace("`", "'") for p in e["pieces"]) + "`", ""]
    with open(report_dir / "gate2_tokenizer.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[gate2] {'PASS' if passed else 'FAIL'}; report: {report_dir / 'gate2_tokenizer.md'}")
    if not passed:
        print("[gate2] If parity or partial-byte checks fail: rebalance tokenizer.sample_bytes_per_language "
              "for the weak language, retrain, and rerun this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
