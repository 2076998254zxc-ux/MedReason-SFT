from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

QA_METRICS = {
    "exact_match": "Exact Match",
    "contains_reference": "Reference Containment",
    "token_recall": "Token Recall",
    "normalized_exact_match": "Normalized Exact",
    "choice_accuracy": "Choice Accuracy",
    "numeric_tolerance_accuracy": "Numeric Tolerance",
}
APPLICABLE_DENOMINATORS = {
    "choice_accuracy": "choice_applicable",
    "numeric_tolerance_accuracy": "numeric_applicable",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in QA_METRICS}
    scores = {}
    for key in QA_METRICS:
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


def load_training_history(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def training_points(history: list[dict[str, Any]], metric: str) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    for row in history:
        if metric in row and "step" in row:
            steps.append(int(row["step"]))
            values.append(float(row[metric]))
    return steps, values


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed = []
    for index in range(len(values)):
        left = max(0, index - window + 1)
        smoothed.append(sum(values[left : index + 1]) / (index - left + 1))
    return smoothed


def plot_training(history: list[dict[str, Any]], output_file: Path, smooth_window: int) -> None:
    import matplotlib.pyplot as plt

    train_steps, train_loss = training_points(history, "loss")
    eval_steps, eval_loss = training_points(history, "eval_loss")
    train_loss = moving_average(train_loss, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if train_steps:
        ax.plot(train_steps, train_loss, label=f"train loss, moving avg={smooth_window}", linewidth=1.8)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="eval loss", marker="o", linewidth=1.8)
    ax.set_title("Verifier-Weighted SFT Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_qa(baseline_rows: list[dict[str, Any]], finetuned_rows: list[dict[str, Any]], output_file: Path) -> None:
    import matplotlib.pyplot as plt

    baseline = mean_scores(baseline_rows)
    finetuned = mean_scores(finetuned_rows)
    labels = list(QA_METRICS.values())
    keys = list(QA_METRICS)
    x_positions = range(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar([x - width / 2 for x in x_positions], [baseline[key] * 100 for key in keys], width, label="baseline")
    ax.bar([x + width / 2 for x in x_positions], [finetuned[key] * 100 for key in keys], width, label="fine-tuned")
    ax.set_title("QA Metrics Before vs After Weighted SFT")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for index, key in enumerate(keys):
        delta = (finetuned[key] - baseline[key]) * 100
        ax.text(index, max(baseline[key], finetuned[key]) * 100 + 2, f"{delta:+.2f} pp", ha="center", fontsize=9)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def recommendation(history: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], finetuned_rows: list[dict[str, Any]]) -> str:
    _, train_loss = training_points(history, "loss")
    _, eval_loss = training_points(history, "eval_loss")
    baseline = mean_scores(baseline_rows)
    finetuned = mean_scores(finetuned_rows)
    qa_delta = finetuned.get("choice_accuracy", finetuned["exact_match"]) - baseline.get("choice_accuracy", baseline["exact_match"])

    notes = []
    if len(eval_loss) >= 2 and eval_loss[-1] > min(eval_loss[:-1]) * 1.03:
        notes.append("eval loss has started to rebound; prefer lowering learning rate or stopping earlier.")
    elif len(eval_loss) >= 2 and eval_loss[-1] < eval_loss[0] * 0.98:
        notes.append("eval loss is still improving; a second epoch or max_length=4096 can be tested next.")
    if len(train_loss) >= 10 and train_loss[-1] > train_loss[0] * 0.98:
        notes.append("train loss is almost flat; try learning_rate=2e-5 or lora_r=32 if GPU memory allows.")
    if qa_delta < 0:
        notes.append("QA task-aware accuracy dropped; reduce epochs or learning rate and inspect generated answer style.")
    elif qa_delta < 0.01:
        notes.append("QA task-aware accuracy barely moved; keep epoch=1 but test learning_rate=2e-5.")
    else:
        notes.append("QA task-aware accuracy improved; keep this run as the stable baseline before trying larger settings.")
    return "\n".join(f"- {note}" for note in dict.fromkeys(notes))


def write_report(args: argparse.Namespace, history: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], finetuned_rows: list[dict[str, Any]]) -> None:
    report = f"""# Experiment Plots

## Stable Starting Config

- Model: Qwen2.5-7B-Instruct
- Method: QLoRA 4-bit NF4
- Max length: 2048
- Epochs: 1
- Learning rate: 1e-5
- LoRA: r=16, alpha=32, dropout=0.05
- Batch: per_device_train_batch_size=1, gradient_accumulation_steps=16
- Scheduler: cosine, warmup_ratio=0.03

## Generated Figures

- Training curve: `{args.training_plot}`
- QA comparison: `{args.qa_plot}`

## Next-Run Recommendation

{recommendation(history, baseline_rows, finetuned_rows)}
"""
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    history = load_training_history(args.training_log)
    baseline_rows = read_jsonl(args.baseline_qa)
    finetuned_rows = read_jsonl(args.finetuned_qa)
    plot_training(history, args.training_plot, args.smooth_window)
    plot_qa(baseline_rows, finetuned_rows, args.qa_plot)
    write_report(args, history, baseline_rows, finetuned_rows)


def self_test() -> None:
    history = [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}, {"step": 2, "eval_loss": 1.5}]
    assert training_points(history, "loss") == ([1, 2], [2.0, 1.0])
    assert moving_average([1, 3, 5], 2) == [1.0, 2.0, 4.0]
    baseline = [{"scores": {"exact_match": 0, "contains_reference": 0, "token_recall": 0.5, "choice_accuracy": 0, "choice_applicable": 1}}]
    finetuned = [{"scores": {"exact_match": 1, "contains_reference": 1, "token_recall": 1.0, "choice_accuracy": 1, "choice_applicable": 1}}]
    assert "improved" in recommendation(history, baseline, finetuned)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training and QA metrics for parameter tuning.")
    parser.add_argument("--training-log", type=Path, default=OUTPUTS_DIR / "stage3_qwen25_7b_weighted_sft" / "training_log_history.jsonl")
    parser.add_argument("--baseline-qa", type=Path, default=OUTPUTS_DIR / "qa_eval_baseline_predictions.jsonl")
    parser.add_argument("--finetuned-qa", type=Path, default=OUTPUTS_DIR / "qa_eval_predictions.jsonl")
    parser.add_argument("--training-plot", type=Path, default=OUTPUTS_DIR / "plots" / "weighted_sft_loss.png")
    parser.add_argument("--qa-plot", type=Path, default=OUTPUTS_DIR / "plots" / "qa_before_after.png")
    parser.add_argument("--report-file", type=Path, default=OUTPUTS_DIR / "plots" / "experiment_plot_report.md")
    parser.add_argument("--smooth-window", type=int, default=5)
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
