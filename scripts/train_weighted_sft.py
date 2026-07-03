from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_weight(row: dict[str, Any]) -> float:
    metadata = row.get("metadata") or {}
    verifier = metadata.get("verifier_scores") or {}
    value = metadata.get("train_weight", verifier.get("train_weight", 1.0))
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 1.0


def fallback_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    chunks = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        chunks.append(f"<|{role}|>\n{content}")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "\n".join(chunks).strip() + "\n"


def render_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return fallback_chat_template(messages, add_generation_prompt=add_generation_prompt)


def last_assistant_index(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("SFT row must contain an assistant message")


def tokenize_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    messages = row["messages"]
    assistant_index = last_assistant_index(messages)
    prompt_messages = messages[:assistant_index]
    supervised_messages = messages[: assistant_index + 1]

    full_text = render_chat(tokenizer, supervised_messages, add_generation_prompt=False)
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
    prompt = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)

    input_ids = full["input_ids"]
    prompt_len = min(len(prompt["input_ids"]), len(input_ids))
    labels = list(input_ids)
    labels[:prompt_len] = [-100] * prompt_len
    loss_tokens = sum(1 for label in labels if label != -100)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "sample_weight": train_weight(row),
        "num_loss_tokens": loss_tokens,
    }


@dataclass
class WeightedDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        sample_weights = torch.tensor([feature.pop("sample_weight") for feature in features], dtype=torch.float32)
        for feature in features:
            feature.pop("num_loss_tokens", None)

        labels = [feature["labels"] for feature in features]
        batch = self.tokenizer.pad(
            [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]} for feature in features],
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["sample_weight"] = sample_weights
        return batch


def import_training_stack() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def make_weighted_trainer(base_trainer: Any, torch: Any, loss_normalization: str) -> type:
    class WeightedSFTTrainer(base_trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any):
            labels = inputs.pop("labels")
            sample_weight = inputs.pop("sample_weight").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels.ne(-100).float()
            safe_labels = shift_labels.masked_fill(shift_labels.eq(-100), 0)

            token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                safe_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)
            weights = sample_weight.view(-1, 1).to(token_loss.dtype)
            weighted_loss = token_loss * mask * weights

            if loss_normalization == "weighted_tokens":
                denominator = (mask * weights).sum().clamp_min(1.0)
            else:
                denominator = mask.sum().clamp_min(1.0)
            loss = weighted_loss.sum() / denominator
            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer


def build_dataset(rows: list[dict[str, Any]], tokenizer: Any, max_length: int, dataset_cls: Any) -> Any:
    dataset = dataset_cls.from_list(rows)
    tokenized = dataset.map(
        lambda row: tokenize_row(row, tokenizer, max_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing weighted SFT rows",
    )
    return tokenized.filter(lambda row: row["num_loss_tokens"] > 0, desc="Dropping prompt-only rows")


def training_arguments_kwargs(args: argparse.Namespace, training_arguments_cls: Any) -> dict[str, Any]:
    params = set(inspect.signature(training_arguments_cls.__init__).parameters)
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": args.report_to,
        "optim": args.optim,
    }
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"
    return kwargs


def trainer_init_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    params = set(inspect.signature(trainer_cls.__init__).parameters)
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}


def load_model_and_tokenizer(args: argparse.Namespace, stack: dict[str, Any]) -> tuple[Any, Any]:
    torch = stack["torch"]
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.method == "qlora":
        quantization_config = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = stack["AutoModelForCausalLM"].from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=dtype if args.method != "qlora" else None,
        device_map="auto",
    )
    model.config.use_cache = False

    if args.method == "qlora":
        model = stack["prepare_model_for_kbit_training"](model)
    if args.method in {"lora", "qlora"}:
        lora_config = stack["LoraConfig"](
            task_type=stack["TaskType"].CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
        )
        model = stack["get_peft_model"](model, lora_config)
        model.print_trainable_parameters()
    return model, tokenizer


def write_training_card(args: argparse.Namespace, train_rows: int, valid_rows: int) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "model_name_or_path": args.model_name_or_path,
        "method": args.method,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "loss": "assistant_token_cross_entropy * metadata.train_weight",
        "loss_normalization": args.loss_normalization,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
    }
    (args.output_dir / "weighted_sft_config.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_log_history(output_dir: Path, log_history: list[dict[str, Any]], eval_metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_log_history.jsonl").open("w", encoding="utf-8") as f:
        for row in log_history:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "final_eval_metrics.json").write_text(
        json.dumps(eval_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    stack = import_training_stack()
    stack["set_seed"](args.seed)
    model, tokenizer = load_model_and_tokenizer(args, stack)
    train_rows = read_jsonl(args.train_file)
    valid_rows = read_jsonl(args.valid_file)
    train_dataset = build_dataset(train_rows, tokenizer, args.max_length, stack["Dataset"])
    valid_dataset = build_dataset(valid_rows, tokenizer, args.max_length, stack["Dataset"])

    trainer_cls = make_weighted_trainer(stack["Trainer"], stack["torch"], args.loss_normalization)
    trainer = trainer_cls(
        model=model,
        args=stack["TrainingArguments"](**training_arguments_kwargs(args, stack["TrainingArguments"])),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=WeightedDataCollator(tokenizer),
        **trainer_init_kwargs(trainer_cls, tokenizer),
    )
    write_training_card(args, len(train_dataset), len(valid_dataset))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    eval_metrics = trainer.evaluate()
    write_log_history(args.output_dir, trainer.state.log_history, eval_metrics)


def self_test() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "metadata": {"train_weight": 0.42},
    }
    assert train_weight(row) == 0.42
    assert last_assistant_index(row["messages"]) == 2
    assert "<|assistant|>" in fallback_chat_template(row["messages"][:2], add_generation_prompt=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train verifier-weighted SFT with LoRA/QLoRA.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-file", type=Path, default=PROCESSED_DIR / "sft_train.jsonl")
    parser.add_argument("--valid-file", type=Path, default=PROCESSED_DIR / "sft_valid.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "stage3_weighted_sft")
    parser.add_argument("--method", choices=["qlora", "lora", "full"], default="qlora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--loss-normalization", choices=["weighted_tokens", "tokens"], default="weighted_tokens")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--self-test", action="store_true")
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
