from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

METRIC_LABELS = {
    "exact_match": "Exact Match",
    "contains_reference": "Reference Containment",
    "token_recall": "Token Recall",
    "normalized_exact_match": "Normalized Exact Match",
    "choice_accuracy": "Choice Accuracy",
    "numeric_tolerance_accuracy": "Numeric Tolerance Accuracy",
}
APPLICABLE_DENOMINATORS = {
    "choice_accuracy": "choice_applicable",
    "numeric_tolerance_accuracy": "numeric_applicable",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in METRIC_LABELS}
    scores = {}
    for key in METRIC_LABELS:
        denominator_key = APPLICABLE_DENOMINATORS.get(key)
        if denominator_key:
            applicable = [row for row in rows if float(row.get("scores", {}).get(denominator_key, 0.0)) > 0]
            scores[key] = (
                mean(float(row.get("scores", {}).get(key, 0.0)) for row in applicable)
                if applicable
                else 0.0
            )
        else:
            scores[key] = mean(float(row.get("scores", {}).get(key, 0.0)) for row in rows)
    return scores


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp_delta(before: float, after: float) -> str:
    sign = "+" if after >= before else ""
    return f"{sign}{(after - before) * 100:.2f} pp"


def relative_delta(before: float, after: float) -> str:
    if before <= 0:
        return "n/a"
    sign = "+" if after >= before else ""
    return f"{sign}{((after - before) / before) * 100:.2f}%"


def table_row(metric: str, baseline: dict[str, float], finetuned: dict[str, float]) -> str:
    before = baseline[metric]
    after = finetuned[metric]
    return (
        f"| {METRIC_LABELS[metric]} | {pct(before)} | {pct(after)} | "
        f"{pp_delta(before, after)} | {relative_delta(before, after)} |"
    )


def highlight_sentence(args: argparse.Namespace, baseline: dict[str, float], finetuned: dict[str, float], count: int) -> str:
    primary_before = baseline[args.primary_metric]
    primary_after = finetuned[args.primary_metric]
    secondary_before = baseline[args.secondary_metric]
    secondary_after = finetuned[args.secondary_metric]
    return (
        f"- **自动化评测与量化突破**：基于 `{args.benchmark_name}` 构建可复现评测闭环，"
        f"对 baseline 与 verifier-weighted QLoRA 模型进行双重对齐评测。"
        f"在 {count} 条测试样本上，{METRIC_LABELS[args.primary_metric]} "
        f"由 **{pct(primary_before)}** 提升至 **{pct(primary_after)}**"
        f"（{pp_delta(primary_before, primary_after)}）；"
        f"{METRIC_LABELS[args.secondary_metric]} 由 **{pct(secondary_before)}** "
        f"提升至 **{pct(secondary_after)}**（{pp_delta(secondary_before, secondary_after)}），"
        f"增强了短答案医学问答与复杂病例分析的可验证可靠性。"
    )


def report_markdown(args: argparse.Namespace, baseline_rows: list[dict[str, Any]], finetuned_rows: list[dict[str, Any]]) -> str:
    baseline = mean_scores(baseline_rows)
    finetuned = mean_scores(finetuned_rows)
    count = min(len(baseline_rows), len(finetuned_rows))
    lines = [
        "# Final Quantitative Metrics",
        "",
        "## Result Highlight",
        "",
        highlight_sentence(args, baseline, finetuned, count),
        "",
        "## QA Benchmark",
        "",
        f"- Benchmark: `{args.benchmark_name}`",
        f"- Baseline file: `{args.baseline_qa}`",
        f"- Fine-tuned file: `{args.finetuned_qa}`",
        f"- Compared examples: {count}",
        "",
        "| Metric | Baseline | Fine-tuned | Absolute Delta | Relative Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRIC_LABELS:
        lines.append(table_row(metric, baseline, finetuned))
    lines.extend(
        [
            "",
            "## Reporting Notes",
            "",
            "- Exact Match is strict and may understate semantically correct medical answers.",
            "- Reference Containment and Token Recall are lexical smoke-test metrics, not clinical correctness labels.",
            "- For final claims, pair these automatic metrics with a small human review set for medical safety and factuality.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    baseline_rows = read_jsonl(args.baseline_qa)
    finetuned_rows = read_jsonl(args.finetuned_qa)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(report_markdown(args, baseline_rows, finetuned_rows), encoding="utf-8")


def self_test() -> None:
    baseline_rows = [
        {"scores": {"exact_match": 0, "contains_reference": 0, "token_recall": 0.5}},
        {"scores": {"exact_match": 1, "contains_reference": 1, "token_recall": 1.0}},
    ]
    finetuned_rows = [
        {"scores": {"exact_match": 1, "contains_reference": 1, "token_recall": 1.0}},
        {"scores": {"exact_match": 1, "contains_reference": 1, "token_recall": 1.0}},
    ]
    baseline = mean_scores(baseline_rows)
    finetuned = mean_scores(finetuned_rows)
    assert pct(baseline["exact_match"]) == "50.00%"
    assert pp_delta(baseline["exact_match"], finetuned["exact_match"]) == "+50.00 pp"
    args = argparse.Namespace(
        benchmark_name="qa_test",
        primary_metric="exact_match",
        secondary_metric="contains_reference",
        baseline_qa="baseline.jsonl",
        finetuned_qa="finetuned.jsonl",
    )
    assert "由 **50.00%** 提升至 **100.00%**" in highlight_sentence(args, baseline, finetuned, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline vs fine-tuned QA metrics for final reporting.")
    parser.add_argument("--baseline-qa", type=Path, default=OUTPUTS_DIR / "qa_eval_baseline_predictions.jsonl")
    parser.add_argument("--finetuned-qa", type=Path, default=OUTPUTS_DIR / "qa_eval_predictions.jsonl")
    parser.add_argument("--output-file", type=Path, default=OUTPUTS_DIR / "final_quantitative_metrics.md")
    parser.add_argument("--benchmark-name", default="qa_test")
    parser.add_argument("--primary-metric", choices=list(METRIC_LABELS), default="choice_accuracy")
    parser.add_argument("--secondary-metric", choices=list(METRIC_LABELS), default="numeric_tolerance_accuracy")
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
