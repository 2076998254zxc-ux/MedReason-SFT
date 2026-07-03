from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

CONVERSATIONAL_MARKERS = (
    "pretty",
    "somehow",
    "clicks into place",
    "i'd bet",
    "first thought",
    "right?",
    "嗯",
    "大概",
    "可能",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_score(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def step_flags(step: dict[str, Any]) -> list[str]:
    content = str(step.get("content", ""))
    lowered = content.lower()
    flags = []
    if step.get("step_type") == "evidence_mapping":
        flags.append("generic_evidence_mapping")
    if step.get("evidence") == "derived_from_original_cot":
        flags.append("span_not_aligned")
    if len(content) < 20:
        flags.append("very_short_step")
    if len(content) > 600:
        flags.append("very_long_step")
    if any(marker in lowered for marker in CONVERSATIONAL_MARKERS):
        flags.append("conversational_style")
    if step.get("validity") != "valid":
        flags.append("non_valid_step")
    return flags


def row_review_reasons(row: dict[str, Any]) -> list[str]:
    steps = row.get("steps") or []
    flags = Counter(flag for step in steps for flag in step_flags(step))
    reasons = []
    if flags["generic_evidence_mapping"] / max(len(steps), 1) >= 0.6:
        reasons.append("too_many_generic_step_types")
    if flags["span_not_aligned"]:
        reasons.append("needs_evidence_span_alignment")
    if flags["conversational_style"]:
        reasons.append("conversational_or_uncertain_style")
    if len(steps) < 3:
        reasons.append("too_few_steps")
    return reasons


def analyze(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_type_counts = Counter()
    validity_counts = Counter()
    evidence_counts = Counter()
    flag_counts = Counter()
    language_counts = Counter()
    step_counts = []
    review_pool = []

    for row in rows:
        metadata = row.get("metadata") or {}
        language_counts[metadata.get("language", "unknown")] += 1
        steps = row.get("steps") or []
        step_counts.append(len(steps))
        for step in steps:
            step_type_counts[step.get("step_type", "unknown")] += 1
            validity_counts[step.get("validity", "unknown")] += 1
            evidence_counts[step.get("evidence", "unknown")] += 1
            flag_counts.update(step_flags(step))
        reasons = row_review_reasons(row)
        if reasons:
            review_pool.append(
                {
                    "id": row.get("id"),
                    "source_id": row.get("source_id"),
                    "question": row.get("question"),
                    "review_reasons": reasons,
                    "steps": steps,
                    "metadata": metadata,
                }
            )

    summary = {
        "rows": len(rows),
        "steps": sum(step_counts),
        "avg_steps_per_row": round(mean(step_counts), 2) if step_counts else 0,
        "step_types": step_type_counts,
        "validity": validity_counts,
        "evidence": evidence_counts,
        "step_flags": flag_counts,
        "languages": language_counts,
        "review_pool": len(review_pool),
    }
    return summary, review_pool


def bullets(counter: Counter) -> str:
    if not counter:
        return "- none\n"
    return "".join(f"- {key}: {value}\n" for key, value in counter.most_common())


def report_markdown(summary: dict[str, Any], audit_count: int) -> str:
    return f"""# Process Supervision Analysis

## Summary

- Rows: {summary['rows']}
- Steps: {summary['steps']}
- Average steps per row: {summary['avg_steps_per_row']}
- Rows recommended for review: {summary['review_pool']}
- Audit sample exported: {audit_count}

## Step Types

{bullets(summary['step_types'])}
## Validity

{bullets(summary['validity'])}
## Evidence Mapping

{bullets(summary['evidence'])}
## Step-Level Flags

{bullets(summary['step_flags'])}
## Languages

{bullets(summary['languages'])}
## Recommendation

Use this file as a seed PRM analysis set for now. Before formal PRM training, add reviewed invalid or partially valid negative steps and replace generic evidence labels with source-aligned spans where possible.
"""


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input_file)
    summary, review_pool = analyze(rows)
    review_pool = sorted(review_pool, key=lambda row: stable_score(str(row.get("id"))))
    audit_sample = review_pool[: args.audit_size]
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(report_markdown(summary, len(audit_sample)), encoding="utf-8")
    write_jsonl(args.audit_file, audit_sample)


def self_test() -> None:
    rows = [
        {
            "id": "p1",
            "metadata": {"language": "en"},
            "steps": [
                {"step_type": "evidence_mapping", "content": "First thought, this is likely X, right?", "evidence": "derived_from_original_cot", "validity": "valid"},
                {"step_type": "conclusion_generation", "content": "Therefore X.", "evidence": "derived_from_original_cot", "validity": "valid"},
            ],
        }
    ]
    summary, review_pool = analyze(rows)
    assert summary["rows"] == 1
    assert review_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze process-supervision seed data before PRM use.")
    parser.add_argument("--input-file", type=Path, default=PROCESSED_DIR / "process_supervision.jsonl")
    parser.add_argument("--report-file", type=Path, default=OUTPUTS_DIR / "process_supervision_analysis.md")
    parser.add_argument("--audit-file", type=Path, default=OUTPUTS_DIR / "process_supervision_audit_pool.jsonl")
    parser.add_argument("--audit-size", type=int, default=200)
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
