# AutoDL 阶段 3-5 最稳实验线路

目标：使用 Qwen2.5-7B-Instruct + QLoRA + 自定义 Trainer，严格执行 `assistant_token_loss * metadata.train_weight`，并输出训练曲线与 QA 前后对比图，用于下一轮调参。

## 1. 推荐首轮配置

首轮不要贪大，先跑稳闭环：

- model：`Qwen/Qwen2.5-7B-Instruct` 或 AutoDL 本地模型路径
- method：QLoRA 4-bit NF4
- max_length：2048
- epochs：1
- learning_rate：1e-5
- lora_r：16
- lora_alpha：32
- lora_dropout：0.05
- per_device_train_batch_size：1
- gradient_accumulation_steps：16
- scheduler：cosine
- warmup_ratio：0.03
- loss_normalization：weighted_tokens

这组配置最稳，适合先拿到 baseline、训练曲线、QA 指标和最终可量化结果。

## 2. 解压项目

```bash
cd /root
mkdir -p project
unzip /root/autodl-tmp/autodl_weighted_sft_payload.zip -d /root/project
cd /root/project
```

检查关键文件：

```bash
ls scripts/train_weighted_sft.py
ls scripts/evaluate_qa.py
ls scripts/plot_experiment_metrics.py
ls data/processed/sft_train.jsonl
ls data/processed/sft_valid.jsonl
ls data/processed/qa_test.jsonl
```

## 3. 安装依赖

```bash
cd /root/project
python -m pip install -U pip
python -m pip install -r requirements-autodl.txt
python -m pip install -U "huggingface_hub[cli]"
```

使用 Hugging Face 镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 4. 下载基座模型

```bash
mkdir -p /root/autodl-tmp/models

hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /root/autodl-tmp/models/qwen2.5-7b-instruct \
  --local-dir-use-symlinks False
```

后续统一使用本地路径：

```bash
MODEL_PATH=/root/autodl-tmp/models/qwen2.5-7b-instruct
```

如果该模型 ID 下载失败，说明仓库名需要换成你实际使用的 Qwen2.5 模型名；后续命令只要替换 `MODEL_PATH` 即可。

## 5. 脚本自检

```bash
python scripts/train_weighted_sft.py --self-test
python scripts/evaluate_qa.py --self-test --model-name-or-path dummy
python scripts/analyze_process_supervision.py --self-test
python scripts/summarize_final_metrics.py --self-test
python scripts/plot_experiment_metrics.py --self-test
```

## 6. Baseline QA 评测

```bash
python scripts/evaluate_qa.py \
  --model-name-or-path "$MODEL_PATH" \
  --input-file data/processed/qa_test.jsonl \
  --output-file outputs/qa_eval_baseline_predictions.jsonl \
  --report-file outputs/qa_eval_baseline_report.md \
  --load-in-4bit
```

## 7. Weighted SFT 训练

建议用 tmux：

```bash
tmux new -s weighted_sft
```

训练命令：

```bash
python scripts/train_weighted_sft.py \
  --model-name-or-path "$MODEL_PATH" \
  --method qlora \
  --max-length 2048 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --eval-steps 200 \
  --save-steps 200 \
  --bf16 \
  --output-dir outputs/stage3_qwen25_7b_weighted_sft
```

训练结束后会生成：

- `outputs/stage3_qwen25_7b_weighted_sft/final`
- `outputs/stage3_qwen25_7b_weighted_sft/training_log_history.jsonl`
- `outputs/stage3_qwen25_7b_weighted_sft/final_eval_metrics.json`
- `outputs/stage3_qwen25_7b_weighted_sft/weighted_sft_config.json`

## 8. 微调后 QA 评测

```bash
python scripts/evaluate_qa.py \
  --model-name-or-path "$MODEL_PATH" \
  --adapter-path outputs/stage3_qwen25_7b_weighted_sft/final \
  --input-file data/processed/qa_test.jsonl \
  --output-file outputs/qa_eval_predictions.jsonl \
  --report-file outputs/qa_eval_report.md \
  --load-in-4bit
```

## 9. 生成量化结果

```bash
python scripts/summarize_final_metrics.py \
  --baseline-qa outputs/qa_eval_baseline_predictions.jsonl \
  --finetuned-qa outputs/qa_eval_predictions.jsonl \
  --benchmark-name qa_test \
  --output-file outputs/final_quantitative_metrics.md
```

## 10. 生成可视化图像

```bash
python scripts/plot_experiment_metrics.py \
  --training-log outputs/stage3_qwen25_7b_weighted_sft/training_log_history.jsonl \
  --baseline-qa outputs/qa_eval_baseline_predictions.jsonl \
  --finetuned-qa outputs/qa_eval_predictions.jsonl \
  --training-plot outputs/plots/weighted_sft_loss.png \
  --qa-plot outputs/plots/qa_before_after.png \
  --report-file outputs/plots/experiment_plot_report.md
```

你要重点看：

- `outputs/plots/weighted_sft_loss.png`
- `outputs/plots/qa_before_after.png`
- `outputs/plots/experiment_plot_report.md`

## 11. 看图调参规则

- train loss 下降，eval loss 也下降：模型还在学，可以尝试 `epochs=2` 或 `max_length=4096`
- train loss 下降，eval loss 反弹：开始过拟合，降低 learning rate 或提前停止
- train loss 基本不动：学习率可能偏低，可以试 `learning_rate=2e-5`
- QA 指标下降：模型输出风格可能变长或偏离短答案，优先回退到 `epochs=1`、`learning_rate=1e-5`
- QA 指标提升但 eval loss 一般：先保留该模型作为稳定 baseline，再单独尝试更大 max_length

建议每次只改一个参数，不要同时改 learning rate、epoch 和 max_length。

## 12. 推荐第二轮候选配置

如果首轮图像健康，优先尝试下面三种之一：

方案 A：更强学习率

```bash
--learning-rate 2e-5 --epochs 1 --max-length 2048
```

方案 B：更长上下文

```bash
--learning-rate 1e-5 --epochs 1 --max-length 4096
```

方案 C：更充分训练

```bash
--learning-rate 1e-5 --epochs 2 --max-length 2048
```

最稳顺序：先 A，再 C，最后 B。
