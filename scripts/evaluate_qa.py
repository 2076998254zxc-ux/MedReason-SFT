from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
OPTION_RE = re.compile(r"(?im)^\s*([A-E])\s*[\.\)]\s+")
ANSWER_CHOICE_RE = re.compile(
    r"(?i)(?:correct\s+answer\s+is|answer\s+is|option|选项|答案是|答案)\s*[:：]?\s*([A-E])\b|^\s*([A-E])\s*[\.\)]?\b"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")


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


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[\s\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def canonical_short_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"^(?:the\s+)?(?:correct\s+)?answer\s+(?:is|:)\s*", "", text)
    text = re.sub(r"^option\s+", "", text)
    text = re.sub(r"^[a-e]\s*[\.\)]\s*", "", text)
    return normalize_answer(text)


def question_choices(question: str) -> set[str]:
    return {match.group(1).upper() for match in OPTION_RE.finditer(question)}


def extract_choice(text: str) -> str | None:
    match = ANSWER_CHOICE_RE.search(unicodedata.normalize("NFKC", text))
    if not match:
        return None
    return (match.group(1) or match.group(2)).upper()


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in NUMBER_RE.finditer(unicodedata.normalize("NFKC", text)):
        try:
            numbers.append(float(match.group(0)))
        except ValueError:
            pass
    return numbers


def numeric_match(prediction: str, reference: str, rel_tol: float = 0.03, abs_tol: float = 0.05) -> tuple[float, float]:
    ref_numbers = extract_numbers(reference)
    pred_numbers = extract_numbers(prediction)
    if not ref_numbers:
        return 0.0, 0.0
    if not pred_numbers:
        return 1.0, 0.0
    for ref in ref_numbers:
        for pred in pred_numbers:
            tolerance = max(abs_tol, abs(ref) * rel_tol)
            if abs(pred - ref) <= tolerance:
                return 1.0, 1.0
    return 1.0, 0.0


def score_prediction(prediction: str, reference: str, question: str = "") -> dict[str, float]:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    exact = float(pred == ref)
    contains = float(bool(ref) and ref in pred)
    normalized_exact = float(canonical_short_answer(prediction) == canonical_short_answer(reference))
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    common = set(pred_tokens) & set(ref_tokens)
    token_recall = len(common) / max(len(set(ref_tokens)), 1)

    choices = question_choices(question)
    ref_choice = extract_choice(reference)
    pred_choice = extract_choice(prediction)
    choice_applicable = float(bool(choices) and ref_choice in choices)
    choice_accuracy = float(choice_applicable and pred_choice == ref_choice)
    numeric_applicable, numeric_accuracy = numeric_match(prediction, reference)

    return {
        "exact_match": exact,
        "contains_reference": contains,
        "token_recall": round(token_recall, 4),
        "normalized_exact_match": normalized_exact,
        "choice_accuracy": choice_accuracy,
        "choice_applicable": choice_applicable,
        "numeric_tolerance_accuracy": numeric_accuracy,
        "numeric_applicable": numeric_applicable,
    }


def qa_messages(question: str, language: str, prompt_style: str = "strict") -> list[dict[str, str]]:
    if language.startswith("zh"):
        system = "你是严谨的医学问答助手。必须严格遵守输出格式。"
        if prompt_style == "strict" and question_choices(question):
            user = f"问题：{question}\n这是选择题。只输出最终选项字母，例如 A、B、C、D 或 E。不要解释。"
        elif prompt_style == "strict":
            user = f"问题：{question}\n如果答案是数值，只输出最终数值和单位；否则只输出最终短答案。不要解释。"
        else:
            user = f"问题：{question}\n请只回答最终短答案。"
    else:
        system = "You are a careful medical QA assistant. Follow the output format exactly."
        if prompt_style == "strict" and question_choices(question):
            user = (
                f"Question: {question}\n"
                "This is a multiple-choice question. Output only the final option letter, such as A, B, C, D, or E. Do not explain."
            )
        elif prompt_style == "strict":
            user = (
                f"Question: {question}\n"
                "If the answer is numeric, output only the final number and unit. Otherwise, output only the concise final answer. Do not explain."
            )
        else:
            user = f"Question: {question}\nAnswer with only the final short answer."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not args.load_in_4bit else None,
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, tokenizer, torch


def generate_answer(model: Any, tokenizer: Any, torch: Any, row: dict[str, Any], args: argparse.Namespace) -> str:
    metadata = row.get("metadata") or {}
    messages = qa_messages(row["question"], metadata.get("language", "en"), args.prompt_style)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def summarize(results: list[dict[str, Any]]) -> str:
    if not results:
        return "# QA Evaluation\n\nNo rows evaluated.\n"
    keys = ["exact_match", "contains_reference", "token_recall", "normalized_exact_match"]
    averages = {
        key: sum(row["scores"][key] for row in results) / len(results)
        for key in keys
    }
    choice_denominator = sum(row["scores"].get("choice_applicable", 0.0) for row in results)
    numeric_denominator = sum(row["scores"].get("numeric_applicable", 0.0) for row in results)
    choice_accuracy = (
        sum(row["scores"].get("choice_accuracy", 0.0) for row in results) / choice_denominator
        if choice_denominator
        else 0.0
    )
    numeric_accuracy = (
        sum(row["scores"].get("numeric_tolerance_accuracy", 0.0) for row in results) / numeric_denominator
        if numeric_denominator
        else 0.0
    )
    lines = [
        "# QA Evaluation",
        "",
        f"- Examples: {len(results)}",
        f"- Exact match: {averages['exact_match']:.4f}",
        f"- Contains reference: {averages['contains_reference']:.4f}",
        f"- Token recall: {averages['token_recall']:.4f}",
        f"- Normalized exact match: {averages['normalized_exact_match']:.4f}",
        f"- Choice accuracy: {choice_accuracy:.4f} ({int(choice_denominator)} applicable)",
        f"- Numeric tolerance accuracy: {numeric_accuracy:.4f} ({int(numeric_denominator)} applicable)",
        "",
        "Exact/contains/token recall are lexical smoke tests, not clinical correctness judgments.",
        "Choice accuracy is computed only for detected multiple-choice rows; numeric tolerance accuracy is computed only for rows with numeric references.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    model, tokenizer, torch = load_model(args)
    rows = read_jsonl(args.input_file)
    if args.limit:
        rows = rows[: args.limit]
    results = []
    for row in rows:
        prediction = generate_answer(model, tokenizer, torch, row, args)
        scores = score_prediction(prediction, row["answer"], row.get("question", ""))
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "prediction": prediction,
                "scores": scores,
                "metadata": row.get("metadata", {}),
            }
        )
    write_jsonl(args.output_file, results)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(summarize(results), encoding="utf-8")


def self_test() -> None:
    assert normalize_answer("The Answer: A.") == "answer"
    scores = score_prediction("D. Mononucleosis", "Mononucleosis")
    assert scores["contains_reference"] == 1.0
    scores = score_prediction(
        "D",
        "D. The first statement is false and the second is true.",
        "A. both true\nB. both false\nC. first true\nD. first false",
    )
    assert scores["choice_accuracy"] == 1.0
    assert scores["choice_applicable"] == 1.0
    scores = score_prediction("5.4 mEq/L", "The median potassium value is 5.45 mEq/L.")
    assert scores["numeric_tolerance_accuracy"] == 1.0
    assert qa_messages("A. x\nB. y", "en")[1]["content"].count("option letter") == 1
    assert qa_messages("x", "en", "concise")[0]["role"] == "system"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate short-answer QA data without mixing it into long SFT.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--input-file", type=Path, default=PROCESSED_DIR / "qa_test.jsonl")
    parser.add_argument("--output-file", type=Path, default=OUTPUTS_DIR / "qa_eval_predictions.jsonl")
    parser.add_argument("--report-file", type=Path, default=OUTPUTS_DIR / "qa_eval_report.md")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt-style", choices=["strict", "concise"], default="strict")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
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
