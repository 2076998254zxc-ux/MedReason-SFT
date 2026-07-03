# Experiment Plots

## Stable Starting Config

- Model: Qwen2.5-8B-Instruct
- Method: QLoRA 4-bit NF4
- Max length: 2048
- Epochs: 1
- Learning rate: 1e-5
- LoRA: r=16, alpha=32, dropout=0.05
- Batch: per_device_train_batch_size=1, gradient_accumulation_steps=16
- Scheduler: cosine, warmup_ratio=0.03

## Generated Figures

- Training curve: `outputs/plots/weighted_sft_loss.png`
- QA comparison: `outputs/plots/qa_before_after_100_v2.png`

## Next-Run Recommendation

- eval loss is still improving; a second epoch or max_length=4096 can be tested next.
- QA task-aware accuracy dropped; reduce epochs or learning rate and inspect generated answer style.
