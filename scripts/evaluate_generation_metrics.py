from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+(?:\.[0-9]+)?")
METRIC_LABELS = {
    "bleu_1": "BLEU-1",
    "bleu_2": "BLEU-2",
    "bleu_3": "BLEU-3",
    "bleu_4": "BLEU-4",
    "gleu": "GLEU",
    "rouge_1": "ROUGE-1",
    "rouge_2": "ROUGE-2",
    "rouge_l": "ROUGE-L",
    "distinct_1": "Distinct-1",
    "distinct_2": "Distinct-2",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return TOKEN_RE.findall(text)


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(max(len(tokens) - n + 1, 0)))


def clipped_precision(candidate: list[str], reference: list[str], n: int) -> float:
    cand = ngrams(candidate, n)
    ref = ngrams(reference, n)
    if not cand:
        return 0.0
    overlap = sum(min(count, ref[gram]) for gram, count in cand.items())
    return overlap / sum(cand.values())


def ngram_recall(candidate: list[str], reference: list[str], n: int) -> float:
    cand = ngrams(candidate, n)
    ref = ngrams(reference, n)
    if not ref:
        return 0.0
    overlap = sum(min(count, cand[gram]) for gram, count in ref.items())
    return overlap / sum(ref.values())


def sentence_bleu(candidate: list[str], reference: list[str], max_n: int) -> float:
    if not candidate or not reference:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        precision = clipped_precision(candidate, reference, n)
        # Smoothing prevents short answers from making BLEU-4 collapse to zero.
        precisions.append(max(precision, 1e-9))
    brevity_penalty = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(len(candidate), 1))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_n)


def sentence_gleu(candidate: list[str], reference: list[str], max_n: int = 4) -> float:
    if not candidate or not reference:
        return 0.0
    scores = []
    for n in range(1, max_n + 1):
        precision = clipped_precision(candidate, reference, n)
        recall = ngram_recall(candidate, reference, n)
        scores.append(min(precision, recall))
    return sum(scores) / len(scores)


def f1(overlap: int, candidate_total: int, reference_total: int) -> float:
    if overlap <= 0 or candidate_total <= 0 or reference_total <= 0:
        return 0.0
    precision = overlap / candidate_total
    recall = overlap / reference_total
    return 2 * precision * recall / (precision + recall)


def rouge_n(candidate: list[str], reference: list[str], n: int) -> float:
    cand = ngrams(candidate, n)
    ref = ngrams(reference, n)
    overlap = sum(min(count, ref[gram]) for gram, count in cand.items())
    return f1(overlap, sum(cand.values()), sum(ref.values()))


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: list[str], reference: list[str]) -> float:
    return f1(lcs_length(candidate, reference), len(candidate), len(reference))


def distinct(candidates: list[list[str]], n: int) -> float:
    all_ngrams = []
    for tokens in candidates:
        all_ngrams.extend(ngrams(tokens, n).elements())
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    candidate_tokens = [tokenize(row.get("prediction", "")) for row in rows]
    reference_tokens = [tokenize(row.get("reference_answer", "")) for row in rows]
    pair_scores = []
    for candidate, reference in zip(candidate_tokens, reference_tokens):
        pair_scores.append(
            {
                "bleu_1": sentence_bleu(candidate, reference, 1),
                "bleu_2": sentence_bleu(candidate, reference, 2),
                "bleu_3": sentence_bleu(candidate, reference, 3),
                "bleu_4": sentence_bleu(candidate, reference, 4),
                "gleu": sentence_gleu(candidate, reference),
                "rouge_1": rouge_n(candidate, reference, 1),
                "rouge_2": rouge_n(candidate, reference, 2),
                "rouge_l": rouge_l(candidate, reference),
            }
        )
    scores = {
        key: mean(row[key] for row in pair_scores) if pair_scores else 0.0
        for key in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "gleu", "rouge_1", "rouge_2", "rouge_l"]
    }
    scores["distinct_1"] = distinct(candidate_tokens, 1)
    scores["distinct_2"] = distinct(candidate_tokens, 2)
    return scores


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp_delta(before: float, after: float) -> str:
    sign = "+" if after >= before else ""
    return f"{sign}{(after - before) * 100:.2f} pp"


def report_single(name: str, scores: dict[str, float], count: int) -> str:
    lines = [
        f"## {name}",
        "",
        f"- Examples: {count}",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for key, label in METRIC_LABELS.items():
        lines.append(f"| {label} | {pct(scores[key])} |")
    return "\n".join(lines)


def report_comparison(
    baseline_name: str,
    baseline_scores: dict[str, float],
    finetuned_name: str,
    finetuned_scores: dict[str, float],
    count: int,
) -> str:
    lines = [
        "## Before vs After",
        "",
        f"- Compared examples: {count}",
        "",
        "| Metric | Baseline | Fine-tuned | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, label in METRIC_LABELS.items():
        lines.append(
            f"| {label} | {pct(baseline_scores[key])} | {pct(finetuned_scores[key])} | {pp_delta(baseline_scores[key], finetuned_scores[key])} |"
        )
    lines.extend(
        [
            "",
            "## Result Sentence",
            "",
            (
                f"HuatuoGPT-style generation metrics show ROUGE-L changing from {pct(baseline_scores['rouge_l'])} "
                f"to {pct(finetuned_scores['rouge_l'])}, GLEU from {pct(baseline_scores['gleu'])} "
                f"to {pct(finetuned_scores['gleu'])}, and Distinct-2 from {pct(baseline_scores['distinct_2'])} "
                f"to {pct(finetuned_scores['distinct_2'])} on {count} evaluated examples."
            ),
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input_file)
    if args.limit:
        rows = rows[: args.limit]
    scores = score_rows(rows)
    report_parts = ["# HuatuoGPT-Style Generation Metrics", "", report_single(args.input_name, scores, len(rows))]
    payload: dict[str, Any] = {args.input_name: {"examples": len(rows), "scores": scores}}

    if args.compare_file:
        compare_rows = read_jsonl(args.compare_file)
        if args.limit:
            compare_rows = compare_rows[: args.limit]
        compare_scores = score_rows(compare_rows)
        report_parts.extend(
            [
                "",
                report_single(args.compare_name, compare_scores, len(compare_rows)),
                "",
                report_comparison(args.input_name, scores, args.compare_name, compare_scores, min(len(rows), len(compare_rows))),
            ]
        )
        payload[args.compare_name] = {"examples": len(compare_rows), "scores": compare_scores}

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text("\n".join(report_parts) + "\n", encoding="utf-8")
    write_json(args.json_file, payload)


def self_test() -> None:
    candidate = tokenize("D. The median is 5.45 mEq/L.")
    reference = tokenize("The median potassium value is 5.45 mEq/L.")
    assert "5.45" in candidate
    assert sentence_bleu(candidate, reference, 1) > 0
    assert rouge_l(candidate, reference) > 0
    rows = [{"prediction": "A", "reference_answer": "A"}, {"prediction": "B C", "reference_answer": "B"}]
    scores = score_rows(rows)
    assert scores["bleu_1"] > 0
    assert scores["distinct_1"] > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HuatuoGPT-style generation metrics from prediction JSONL.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--input-name", default="baseline")
    parser.add_argument("--compare-file", type=Path, default=None)
    parser.add_argument("--compare-name", default="fine_tuned")
    parser.add_argument("--output-file", type=Path, default=OUTPUTS_DIR / "generation_metrics_report.md")
    parser.add_argument("--json-file", type=Path, default=OUTPUTS_DIR / "generation_metrics.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
