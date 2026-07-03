from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
META_DIR = ROOT / "data" / "metadata"
OUTPUTS_DIR = ROOT / "outputs"

HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\f\v]+")
PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)

RAW_FILES = [
    {
        "path": RAW_DIR / "medical-o1-reasoning-SFT" / "medical_o1_sft.json",
        "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "source_subset": "en",
        "fields": {"question": "Question", "cot": "Complex_CoT", "answer": "Response"},
    },
    {
        "path": RAW_DIR / "medical-o1-reasoning-SFT" / "medical_o1_sft_Chinese.json",
        "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "source_subset": "zh",
        "fields": {"question": "Question", "cot": "Complex_CoT", "answer": "Response"},
    },
    {
        "path": RAW_DIR / "medical-o1-reasoning-SFT" / "medical_o1_sft_mix.json",
        "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "source_subset": "en_mix",
        "fields": {"question": "Question", "cot": "Complex_CoT", "answer": "Response"},
    },
    {
        "path": RAW_DIR / "medical-o1-reasoning-SFT" / "medical_o1_sft_mix_Chinese.json",
        "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "source_subset": "zh_mix",
        "fields": {"question": "Question", "cot": "Complex_CoT", "answer": "Response"},
    },
    {
        "path": RAW_DIR / "medical-o1-verifiable-problem" / "medical_o1_verifiable_problem.json",
        "source_dataset": "FreedomIntelligence/medical-o1-verifiable-problem",
        "source_subset": "en",
        "fields": {
            "question": "Open-ended Verifiable Question",
            "answer": "Ground-True Answer",
        },
    },
]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_RE.sub(" ", text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def effective_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def language_of(text: str, source_subset: str) -> str:
    if source_subset in {"zh", "zh_mix", "en_mix"}:
        return source_subset
    # ponytail: cheap language heuristic for stage 1; upgrade to a classifier if mixed-language routing becomes important.
    zh = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = zh + ascii_letters
    if not total:
        return "unknown"
    if zh / total >= 0.3:
        return "zh"
    if ascii_letters / total >= 0.7:
        return "en"
    return "mixed"


def risk_tags(text: str, rules: dict) -> list[str]:
    # ponytail: keyword tags are high-recall labels only; upgrade to a reviewed classifier before using them as safety ground truth.
    lowered = text.lower()
    tags = []
    for tag, words in rules["risk_keywords"].items():
        if any(word.lower() in lowered for word in words):
            tags.append(tag)
    for tag, phrases in rules.get("strict_phrase_tags", {}).items():
        if any(phrase.lower() in lowered for phrase in phrases):
            tags.append(tag)
    if "pregnancy_child" in tags and is_public_health_context(lowered, rules):
        tags.remove("pregnancy_child")
    return tags


def is_public_health_context(lowered_text: str, rules: dict) -> bool:
    return any(term.lower() in lowered_text for term in rules.get("public_health_context_terms", []))


def is_non_medical_or_weak_medical(text: str, rules: dict) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in rules.get("non_medical_or_weak_medical_terms", []))


def is_unreadable(text: str) -> bool:
    if "\ufffd" in text:
        return True
    controls = sum(1 for ch in text if unicodedata.category(ch).startswith("C") and ch not in "\n\t")
    return bool(text) and controls / max(len(text), 1) > 0.02


def normalize_question(text: str) -> str:
    return PUNCT_RE.sub("", unicodedata.normalize("NFKC", text).lower())


def load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def public_rows(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def load_rules() -> dict:
    with (META_DIR / "cleaning_rules.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def build_unified() -> tuple[list[dict], Counter]:
    rows = []
    counts = Counter()
    next_id = 1
    for spec in RAW_FILES:
        records = load_json_list(spec["path"])
        counts[f"raw::{spec['path'].name}"] = len(records)
        for source_index, record in enumerate(records):
            fields = spec["fields"]
            question = clean_text(record.get(fields["question"]))
            answer = clean_text(record.get(fields["answer"]))
            cot = clean_text(record.get(fields["cot"])) if "cot" in fields else ""
            text_for_tags = "\n".join(part for part in [question, answer, cot] if part)
            rows.append(
                {
                    "id": f"medreason_{next_id:06d}",
                    "source_dataset": spec["source_dataset"],
                    "source_file": spec["path"].name,
                    "source_index": source_index,
                    "source_subset": spec["source_subset"],
                    "question": question,
                    "answer": answer,
                    "cot": cot,
                    "language": language_of(question + "\n" + answer, spec["source_subset"]),
                    "task_type": "unknown",
                    "department": "unknown",
                    "risk_tags": [],
                    "quality_flags": [],
                    "metadata": {"has_cot": bool(cot)},
                    "_tag_text": text_for_tags,
                }
            )
            next_id += 1
    return rows, counts


def clean_rows(rows: list[dict], rules: dict) -> tuple[list[dict], Counter]:
    kept = []
    removed = Counter()
    for row in rows:
        question = row["question"]
        answer = row["answer"]
        cot = row["cot"]
        if not question:
            removed["missing_question"] += 1
            continue
        if not answer:
            removed["missing_answer"] += 1
            continue
        if effective_len(question) < rules["min_question_chars"]:
            removed["short_question"] += 1
            continue
        if effective_len(answer) < rules["min_answer_chars"]:
            removed["short_answer"] += 1
            continue
        if normalize_question(question) == normalize_question(answer):
            removed["question_equals_answer"] += 1
            continue
        if is_unreadable(question + answer + cot):
            removed["unreadable_text"] += 1
            continue

        flags = []
        tags = risk_tags(row.pop("_tag_text"), rules)
        if not cot:
            flags.append("missing_cot")
        if "insufficient_information" in tags:
            flags.append("possible_missing_context")
        if is_non_medical_or_weak_medical(question + "\n" + answer + "\n" + cot, rules):
            flags.append("non_medical_or_weak_medical")
        if effective_len(answer) < 10:
            flags.append("short_answer")
        if cot and len(cot) > rules["max_cot_chars"]:
            flags.append("long_cot")

        row["risk_tags"] = tags
        row["quality_flags"] = flags
        kept.append(row)
    return kept, removed


def row_rank(row: dict) -> tuple[int, int, int, int]:
    return (
        1 if row["metadata"]["has_cot"] else 0,
        1 if row["source_dataset"].endswith("medical-o1-reasoning-SFT") else 0,
        -len(row["quality_flags"]),
        min(len(row["answer"]), 2000),
    )


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], Counter]:
    winners = {}
    removed = Counter()
    for row in rows:
        key = normalize_question(row["question"])
        if key not in winners:
            winners[key] = row
            continue
        removed["duplicate_question"] += 1
        if row_rank(row) > row_rank(winners[key]):
            winners[key] = row
    deduped = sorted(winners.values(), key=lambda item: int(item["id"].rsplit("_", 1)[1]))
    return deduped, removed


def summarize_lengths(rows: list[dict], field: str) -> dict:
    lengths = [effective_len(row[field]) for row in rows if row.get(field)]
    if not lengths:
        return {"count": 0, "avg": 0, "max": 0}
    return {
        "count": len(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
        "max": max(lengths),
    }


def make_report(
    raw_rows: list[dict],
    unified_counts: Counter,
    cleaned_rows: list[dict],
    clean_removed: Counter,
    deduped_rows: list[dict],
    dedupe_removed: Counter,
) -> str:
    by_source = Counter(row["source_file"] for row in raw_rows)
    by_language = Counter(row["language"] for row in deduped_rows)
    risk_dist = Counter(tag for row in deduped_rows for tag in row["risk_tags"])
    flag_dist = Counter(flag for row in deduped_rows for flag in row["quality_flags"])
    length_rows = [
        ("question", summarize_lengths(deduped_rows, "question")),
        ("answer", summarize_lengths(deduped_rows, "answer")),
        ("cot", summarize_lengths(deduped_rows, "cot")),
    ]

    def bullets(counter: Counter) -> str:
        if not counter:
            return "- none\n"
        return "".join(f"- {key}: {value}\n" for key, value in counter.most_common())

    length_table = "\n".join(
        f"| {name} | {stats['count']} | {stats['avg']} | {stats['max']} |"
        for name, stats in length_rows
    )
    return f"""# Stage 1 Data Report

Generated at: {datetime.now(timezone.utc).isoformat()}

## Inputs

- Raw records: {len(raw_rows)}
- Cleaned records: {len(cleaned_rows)}
- Deduped records: {len(deduped_rows)}

## Raw Files

{bullets(by_source)}
## Raw Load Counts

{bullets(unified_counts)}
## Removed During Cleaning

{bullets(clean_removed)}
## Removed During Deduplication

{bullets(dedupe_removed)}
## Deduped Language Distribution

{bullets(by_language)}
## Deduped Risk Tag Distribution

{bullets(risk_dist)}
## Deduped Quality Flag Distribution

{bullets(flag_dist)}
## Deduped Length Summary

| field | non-empty count | avg effective chars | max effective chars |
|---|---:|---:|---:|
{length_table}

## Outputs

- data/interim/unified_raw.jsonl
- data/interim/cleaned.jsonl
- data/interim/deduped.jsonl
"""


def run() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(spec["path"]) for spec in RAW_FILES if not spec["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing raw data files:\n" + "\n".join(missing))

    rules = load_rules()
    raw_rows, unified_counts = build_unified()
    write_jsonl(INTERIM_DIR / "unified_raw.jsonl", public_rows(raw_rows))

    cleaned_rows, clean_removed = clean_rows([dict(row) for row in raw_rows], rules)
    write_jsonl(INTERIM_DIR / "cleaned.jsonl", cleaned_rows)

    deduped_rows, dedupe_removed = dedupe_rows(cleaned_rows)
    write_jsonl(INTERIM_DIR / "deduped.jsonl", deduped_rows)

    report = make_report(raw_rows, unified_counts, cleaned_rows, clean_removed, deduped_rows, dedupe_removed)
    (OUTPUTS_DIR / "data_report.md").write_text(report, encoding="utf-8")


def self_test() -> None:
    rules = {
        "min_question_chars": 5,
        "min_answer_chars": 2,
        "max_cot_chars": 20,
        "risk_keywords": {"emergency_triage": ["胸痛"], "pregnancy_child": ["infant"]},
        "strict_phrase_tags": {"insufficient_information": ["as shown in figure"]},
        "public_health_context_terms": ["infant mortality"],
        "non_medical_or_weak_medical_terms": ["gross domestic product"],
    }
    rows = [
        {
            "id": "medreason_000001",
            "source_dataset": "FreedomIntelligence/medical-o1-verifiable-problem",
            "source_file": "a.json",
            "source_index": 0,
            "source_subset": "zh",
            "question": "患者胸痛怎么办？",
            "answer": "及时就医",
            "cot": "",
            "language": "zh",
            "task_type": "unknown",
            "department": "unknown",
            "risk_tags": [],
            "quality_flags": [],
            "metadata": {"has_cot": False},
            "_tag_text": "患者胸痛怎么办？\n及时就医",
        },
        {
            "id": "medreason_000002",
            "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "source_file": "b.json",
            "source_index": 1,
            "source_subset": "zh",
            "question": "患者胸痛怎么办",
            "answer": "胸痛属于高风险症状，应尽快就医。",
            "cot": "识别症状，给出安全建议。",
            "language": "zh",
            "task_type": "unknown",
            "department": "unknown",
            "risk_tags": [],
            "quality_flags": [],
            "metadata": {"has_cot": True},
            "_tag_text": "患者胸痛怎么办\n胸痛属于高风险症状，应尽快就医。",
        },
    ]
    cleaned, removed = clean_rows([dict(row) for row in rows], rules)
    deduped, dedup_removed = dedupe_rows(cleaned)
    assert not removed
    assert dedup_removed["duplicate_question"] == 1
    assert len(deduped) == 1
    assert deduped[0]["id"] == "medreason_000002"
    assert deduped[0]["risk_tags"] == ["emergency_triage"]
    assert clean_text("a\xa0 b") == "a b"
    assert "pregnancy_child" not in risk_tags("infant mortality rate", rules)
    assert "insufficient_information" in risk_tags("diagnosis as shown in figure", rules)
    assert is_non_medical_or_weak_medical("gross domestic product", rules)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        write_jsonl(path, deduped)
        assert path.read_text(encoding="utf-8").count("\n") == 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stage-1 MedReason data files.")
    parser.add_argument("--self-test", action="store_true", help="run the tiny built-in pipeline check")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run()


if __name__ == "__main__":
    main()
