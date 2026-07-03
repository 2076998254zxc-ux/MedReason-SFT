from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
META_DIR = ROOT / "data" / "metadata"
OUTPUTS_DIR = ROOT / "outputs"

LONG_SFT_MIN_ANSWER_CHARS = 80
QA_MAX_ANSWER_CHARS = 80
MIN_PROCESS_STEPS = 2
MIN_VERIFIER_WEIGHT = 0.72

SENTENCE_RE = re.compile(r"(?<=[\u3002\uff01\uff1f.!?])\s+|\n+")
FILLER_PREFIXES = (
    "okay",
    "ok,",
    "alright",
    "let's",
    "so,",
    "so ",
    "\u55ef",
    "\u54e6",
    "\u597d\u5427",
    "\u8ba9\u6211\u4eec",
    "\u6211\u6765",
    "\u6211\u4eec\u6765",
)
WEAK_REASONING_PHRASES = (
    "hmm",
    "i think",
    "i guess",
    "maybe",
    "probably",
    "sort of",
    "kind of",
    "not a great start",
    "let's continue",
    "\u6211\u89c9\u5f97",
    "\u5927\u6982",
    "\u53ef\u80fd",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_row(file, row: dict) -> None:
    file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            write_row(f, row)
            count += 1
    return count


def stable_bucket(row_id: str) -> int:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_name(row: dict) -> str:
    bucket = stable_bucket(row["id"])
    if bucket < 85:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def effective_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def response_language(row: dict) -> str:
    return "zh" if row["language"].startswith("zh") else "en"


def system_prompt(row: dict, safety_rules: dict) -> str:
    return safety_rules["system_prompts"][response_language(row)]


def safety_reminder(row: dict, safety_rules: dict) -> str:
    lang = response_language(row)
    reminder_map = safety_rules["risk_reminders"][lang]
    reminders = [reminder_map[tag] for tag in row["risk_tags"] if tag in reminder_map]
    if reminders:
        return " ".join(dict.fromkeys(reminders))
    return safety_rules["default_reminders"][lang]


def assistant_content(row: dict, safety_rules: dict) -> str:
    answer = row["answer"].strip()
    reminder = safety_reminder(row, safety_rules)
    if reminder in answer:
        return answer
    marker = safety_rules["safety_markers"][response_language(row)]
    return f"{answer}\n\n{marker}: {reminder}"


def quality_score(row: dict) -> float:
    score = 1.0
    if "missing_cot" in row["quality_flags"]:
        score -= 0.12
    if "short_answer" in row["quality_flags"]:
        score -= 0.1
    if "possible_missing_context" in row["quality_flags"]:
        score -= 0.15
    if "non_medical_or_weak_medical" in row["quality_flags"]:
        score -= 0.25
    if effective_len(row["answer"]) < LONG_SFT_MIN_ANSWER_CHARS:
        score -= 0.05
    if row["risk_tags"] and not row["metadata"]["has_cot"]:
        score -= 0.05
    return round(max(score, 0.0), 3)


def quality_band(score: float) -> str:
    if score >= 0.9:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.55:
        return "C"
    return "D"


def answer_consistency_score(row: dict) -> float:
    question = row["question"].lower()
    answer = row["answer"].lower()
    score = 0.74
    if effective_len(row["answer"]) >= 160:
        score += 0.08
    if row["metadata"]["has_cot"]:
        score += 0.08
    if any(marker in question for marker in [" a.", " b.", " c.", " d.", " e.", " a ", " b "]):
        if any(marker in answer for marker in [" a.", " b.", " c.", " d.", " e.", "answer is", "答案是", "选"]):
            score += 0.04
    if "possible_missing_context" in row["quality_flags"]:
        score -= 0.25
    if "non_medical_or_weak_medical" in row["quality_flags"]:
        score -= 0.3
    if effective_len(row["answer"]) < LONG_SFT_MIN_ANSWER_CHARS:
        score -= 0.12
    return round(min(max(score, 0.0), 1.0), 3)


def safety_score(row: dict) -> float:
    score = 1.0
    flags = set(row["quality_flags"])
    if "possible_missing_context" in flags:
        score -= 0.25
    if "non_medical_or_weak_medical" in flags:
        score -= 0.2
    if row["risk_tags"] and not row["metadata"]["has_cot"]:
        score -= 0.1
    if "drug_safety" in row["risk_tags"] and effective_len(row["answer"]) < 80:
        score -= 0.08
    return round(min(max(score, 0.0), 1.0), 3)


def process_supervision_score(row: dict) -> float:
    if not row["metadata"]["has_cot"]:
        return 0.45
    parts = split_cot(row["cot"], limit=8)
    if len(parts) >= 4:
        return 1.0
    if len(parts) >= MIN_PROCESS_STEPS:
        return 0.82
    return 0.58


def verifier_scores(row: dict) -> dict:
    answer_score = answer_consistency_score(row)
    safety = safety_score(row)
    process = process_supervision_score(row)
    quality = quality_score(row)
    weight = 0.35 * answer_score + 0.3 * safety + 0.25 * process + 0.1 * quality
    return {
        "answer_consistency": answer_score,
        "medical_safety": safety,
        "process_supervision": process,
        "base_quality": quality,
        "train_weight": round(min(max(weight, 0.0), 1.0), 3),
    }


def weight_bucket(weight: float) -> str:
    if weight >= 0.9:
        return "0.90-1.00"
    if weight >= 0.8:
        return "0.80-0.89"
    if weight >= MIN_VERIFIER_WEIGHT:
        return f"{MIN_VERIFIER_WEIGHT:.2f}-0.79"
    if weight >= 0.6:
        return "0.60-low"
    return "below-0.60"


def analysis_reasons(row: dict) -> list[str]:
    reasons = []
    flags = set(row["quality_flags"])
    if "possible_missing_context" in flags:
        reasons.append("possible_missing_context")
    if "non_medical_or_weak_medical" in flags:
        reasons.append("non_medical_or_weak_medical")
    if quality_band(quality_score(row)) == "D":
        reasons.append("low_quality")
    if is_long_sft_shape(row) and verifier_scores(row)["train_weight"] < MIN_VERIFIER_WEIGHT:
        reasons.append("low_verifier_weight")
    return reasons


def is_qa_candidate(row: dict) -> bool:
    return effective_len(row["answer"]) < QA_MAX_ANSWER_CHARS


def is_long_sft_shape(row: dict) -> bool:
    return effective_len(row["answer"]) >= LONG_SFT_MIN_ANSWER_CHARS


def is_long_sft_candidate(row: dict) -> bool:
    if not is_long_sft_shape(row):
        return False
    if quality_band(quality_score(row)) not in {"A", "B"}:
        return False
    return verifier_scores(row)["train_weight"] >= MIN_VERIFIER_WEIGHT


def build_sft_row(row: dict, safety_rules: dict) -> dict:
    return {
        "id": row["id"],
        "messages": [
            {"role": "system", "content": system_prompt(row, safety_rules)},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": assistant_content(row, safety_rules)},
        ],
        "metadata": common_metadata(row),
    }


def build_qa_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "metadata": common_metadata(row),
    }


def build_analysis_row(row: dict, reasons: list[str]) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "cot": row["cot"],
        "analysis_reasons": reasons,
        "metadata": common_metadata(row),
    }


def common_metadata(row: dict) -> dict:
    return {
        "source_dataset": row["source_dataset"],
        "source_file": row["source_file"],
        "language": row["language"],
        "risk_tags": row["risk_tags"],
        "quality_flags": row["quality_flags"],
        "has_cot": row["metadata"]["has_cot"],
        "quality_score": quality_score(row),
        "quality_band": quality_band(quality_score(row)),
        "verifier_scores": verifier_scores(row),
        "train_weight": verifier_scores(row)["train_weight"],
        "split": split_name(row),
    }


def holdout_row(row: dict, route: str) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "reference_answer": row["answer"],
        "language": row["language"],
        "risk_tags": row["risk_tags"],
        "quality_score": quality_score(row),
        "quality_band": quality_band(quality_score(row)),
        "source_dataset": row["source_dataset"],
        "route": route,
    }


def split_cot(cot: str, limit: int = 8) -> list[str]:
    # ponytail: still sentence-based; upgrade to reviewed step parsing when labels are used for PRM training.
    parts = [part.strip(" -\t") for part in SENTENCE_RE.split(cot) if is_process_step_content(part.strip(" -\t"))]
    return parts[:limit]


def is_process_step_content(text: str) -> bool:
    lowered = text.lower().strip()
    if len(text) < 12:
        return False
    if lowered.startswith(FILLER_PREFIXES):
        return False
    return not any(phrase in lowered for phrase in WEAK_REASONING_PHRASES)


def infer_step_type(text: str, index: int, total: int) -> str:
    lowered = text.lower()
    if index == total - 1:
        return "conclusion_generation"
    if any(word in lowered for word in ["symptom", "sign", "patient presents", "\u60a3\u8005", "\u75c7\u72b6", "\u4f53\u5f81"]):
        return "symptom_extraction"
    if any(word in lowered for word in ["contraindication", "\u7981\u5fcc"]):
        return "contraindication_check"
    if any(word in lowered for word in ["treatment", "therapy", "management", "\u6cbb\u7597", "\u5904\u7406", "\u7528\u836f"]):
        return "treatment_decision"
    if any(word in lowered for word in ["differential", "distinguish", "\u9274\u522b"]):
        return "differential_diagnosis"
    if any(word in lowered for word in ["mechanism", "because", "due to", "pathophysiology", "\u673a\u5236", "\u56e0\u4e3a", "\u7531\u4e8e"]):
        return "mechanism_reasoning"
    return "evidence_mapping"


def build_process_row(row: dict, safety_rules: dict) -> dict | None:
    if not row["cot"]:
        return None
    parts = split_cot(row["cot"])
    if len(parts) < MIN_PROCESS_STEPS:
        return None
    steps = []
    for index, content in enumerate(parts):
        steps.append(
            {
                "step_id": index + 1,
                "step_type": infer_step_type(content, index, len(parts)),
                "content": content,
                "evidence": "derived_from_original_cot",
                "validity": "valid",
                "error_type": "none",
            }
        )
    if row["risk_tags"]:
        steps.append(
            {
                "step_id": len(steps) + 1,
                "step_type": "safety_reminder",
                "content": safety_reminder(row, safety_rules),
                "evidence": "risk_tags",
                "validity": "valid",
                "error_type": "none",
            }
        )
    return {
        "id": f"process_{row['id'].removeprefix('medreason_')}",
        "source_id": row["id"],
        "question": row["question"],
        "steps": steps,
        "final_answer": row["answer"],
        "metadata": common_metadata(row),
    }


def make_report(stats: dict) -> str:
    def bullets(counter: Counter) -> str:
        if not counter:
            return "- none\n"
        return "".join(f"- {key}: {value}\n" for key, value in counter.most_common())

    return f"""# Stage 2 Data Report

Generated at: {datetime.now(timezone.utc).isoformat()}

## Routed Outputs

- Long SFT train: {stats['sft_splits']['train']}
- Long SFT valid: {stats['sft_splits']['valid']}
- Long SFT test: {stats['sft_splits']['test']}
- QA train: {stats['qa_splits']['train']}
- QA valid: {stats['qa_splits']['valid']}
- QA test: {stats['qa_splits']['test']}
- Holdout eval: {stats['holdout']}
- Process supervision: {stats['process']}
- Analysis pool: {stats['analysis']}

## Routing

{bullets(stats['routes'])}
## Analysis Reasons

{bullets(stats['analysis_reasons'])}
## Quality Bands

{bullets(stats['bands'])}
## Verifier Weights

{bullets(stats['weight_buckets'])}
## Languages

{bullets(stats['languages'])}
## Risk Tags

{bullets(stats['risk_tags'])}
## Process Step Types

{bullets(stats['step_types'])}
## PRM Rejections

{bullets(stats['prm_rejections'])}
"""


def open_outputs() -> dict:
    files = {
        "sft_train": (PROCESSED_DIR / "sft_train.jsonl").open("w", encoding="utf-8"),
        "sft_valid": (PROCESSED_DIR / "sft_valid.jsonl").open("w", encoding="utf-8"),
        "sft_test": (PROCESSED_DIR / "sft_test.jsonl").open("w", encoding="utf-8"),
        "qa_train": (PROCESSED_DIR / "qa_train.jsonl").open("w", encoding="utf-8"),
        "qa_valid": (PROCESSED_DIR / "qa_valid.jsonl").open("w", encoding="utf-8"),
        "qa_test": (PROCESSED_DIR / "qa_test.jsonl").open("w", encoding="utf-8"),
        "process": (PROCESSED_DIR / "process_supervision.jsonl").open("w", encoding="utf-8"),
        "holdout": (PROCESSED_DIR / "holdout_eval.jsonl").open("w", encoding="utf-8"),
        "analysis": (PROCESSED_DIR / "analysis_pool.jsonl").open("w", encoding="utf-8"),
    }
    return files


def run() -> None:
    input_path = INTERIM_DIR / "deduped.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    safety_rules = load_json(META_DIR / "safety_rules.json")
    files = open_outputs()
    stats = {
        "sft_splits": Counter(),
        "qa_splits": Counter(),
        "holdout": 0,
        "process": 0,
        "analysis": 0,
        "routes": Counter(),
        "analysis_reasons": Counter(),
        "bands": Counter(),
        "weight_buckets": Counter(),
        "languages": Counter(),
        "risk_tags": Counter(),
        "step_types": Counter(),
        "prm_rejections": Counter(),
    }
    try:
        for row in read_jsonl(input_path):
            split = split_name(row)
            score = quality_score(row)
            band = quality_band(score)
            verifier = verifier_scores(row)
            stats["bands"][band] += 1
            stats["weight_buckets"][weight_bucket(verifier["train_weight"])] += 1
            stats["languages"][row["language"]] += 1
            stats["risk_tags"].update(row["risk_tags"])

            reasons = analysis_reasons(row)
            if reasons:
                route = "analysis"
                write_row(files["analysis"], build_analysis_row(row, reasons))
                stats["analysis"] += 1
                stats["analysis_reasons"].update(reasons)
            elif is_qa_candidate(row):
                route = "qa"
                write_row(files[f"qa_{split}"], build_qa_row(row))
                stats["qa_splits"][split] += 1
            elif is_long_sft_candidate(row):
                route = "sft"
                write_row(files[f"sft_{split}"], build_sft_row(row, safety_rules))
                stats["sft_splits"][split] += 1
            else:
                route = "analysis"
                write_row(files["analysis"], build_analysis_row(row, ["not_routed_to_sft_or_qa"]))
                stats["analysis"] += 1
                stats["analysis_reasons"]["not_routed_to_sft_or_qa"] += 1

            stats["routes"][route] += 1
            if split == "test":
                write_row(files["holdout"], holdout_row(row, route))
                stats["holdout"] += 1

            if route == "sft" and split != "test" and band in {"A", "B"}:
                process = build_process_row(row, safety_rules)
                if process:
                    write_row(files["process"], process)
                    stats["process"] += 1
                    stats["step_types"].update(step["step_type"] for step in process["steps"])
                else:
                    stats["prm_rejections"]["too_few_valid_steps_or_missing_cot"] += 1
    finally:
        for file in files.values():
            file.close()

    (OUTPUTS_DIR / "stage2_report.md").write_text(make_report(stats), encoding="utf-8")


def self_test() -> None:
    safety_rules = {
        "system_prompts": {"zh": "zh system", "en": "English system prompt"},
        "safety_markers": {"zh": "Safety zh", "en": "Safety note"},
        "default_reminders": {"zh": "Ask a doctor.", "en": "Consult a clinician."},
        "risk_reminders": {
            "zh": {"emergency_triage": "Seek care zh."},
            "en": {"emergency_triage": "Seek urgent care."},
        },
    }
    row = {
        "id": "medreason_000123",
        "source_dataset": "x",
        "source_file": "x.json",
        "language": "en",
        "question": "What should be considered for persistent chest pain?",
        "answer": "Persistent chest pain can be associated with high-risk cardiac or pulmonary disease and should be evaluated promptly in clinical care.",
        "cot": "The patient has persistent chest pain with potentially high-risk features. Chest pain can indicate acute coronary syndrome or other urgent disease. Therefore the answer should emphasize timely medical evaluation.",
        "risk_tags": ["emergency_triage"],
        "quality_flags": [],
        "metadata": {"has_cot": True},
    }
    sft = build_sft_row(row, safety_rules)
    process = build_process_row(row, safety_rules)
    assert len(sft["messages"]) == 3
    assert "Safety note" in sft["messages"][2]["content"]
    assert is_long_sft_candidate(row)
    assert not is_qa_candidate(row)
    assert verifier_scores(row)["train_weight"] >= MIN_VERIFIER_WEIGHT
    weak_row = {**row, "quality_flags": ["non_medical_or_weak_medical"]}
    assert not is_long_sft_candidate(weak_row)
    assert "low_verifier_weight" in analysis_reasons(weak_row)
    assert process is not None
    assert len([step for step in process["steps"] if step["step_type"] != "safety_reminder"]) >= MIN_PROCESS_STEPS
    assert build_process_row({**row, "cot": "Okay. Hmm. Maybe."}, safety_rules) is None
    assert is_qa_candidate({**row, "answer": "Fluids"})
    assert "possible_missing_context" in analysis_reasons({**row, "quality_flags": ["possible_missing_context"]})
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.jsonl"
        assert write_jsonl(path, [sft]) == 1
        assert path.read_text(encoding="utf-8").count("\n") == 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-2 SFT, QA, analysis, and process-supervision data.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run()


if __name__ == "__main__":
    main()
