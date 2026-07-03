# 最终结果量化指标模板

最终展示建议模仿截图里的写法：先说明评测闭环，再给出 benchmark 名称，最后写清楚关键指标从多少提升到多少。

## 推荐主指标

| 模块 | 指标 | 来源 |
|---|---|---|
| Weighted SFT | train loss / eval loss 曲线 | `plot_experiment_metrics.py` |
| Weighted SFT | final eval loss | `final_eval_metrics.json` |
| QA 能力 | Exact Match | `evaluate_qa.py` |
| QA 能力 | Reference Containment | `evaluate_qa.py` |
| QA 能力 | Token Recall | `evaluate_qa.py` |
| 生成质量 | BLEU-1/2/3/4、GLEU、ROUGE-1/2/L、Distinct-1/2 | `evaluate_generation_metrics.py` |
| 过程监督 | 平均 step 数、step type 分布、待复核比例 | `analyze_process_supervision.py` |

## 生成量化段落

```bash
python scripts/summarize_final_metrics.py \
  --baseline-qa outputs/qa_eval_baseline_predictions.jsonl \
  --finetuned-qa outputs/qa_eval_predictions.jsonl \
  --benchmark-name qa_test \
  --output-file outputs/final_quantitative_metrics.md
```

## 生成图像

```bash
python scripts/plot_experiment_metrics.py \
  --training-log outputs/stage3_qwen25_7b_weighted_sft/training_log_history.jsonl \
  --baseline-qa outputs/qa_eval_baseline_predictions.jsonl \
  --finetuned-qa outputs/qa_eval_predictions.jsonl
```

输出：

- `outputs/plots/weighted_sft_loss.png`
- `outputs/plots/qa_before_after.png`
- `outputs/plots/experiment_plot_report.md`

## 生成 HuatuoGPT-style 指标

```bash
python scripts/evaluate_generation_metrics.py \
  --input-file outputs/qa_eval_baseline_predictions_100_strict.jsonl \
  --input-name baseline \
  --compare-file outputs/qa_eval_predictions_100_strict.jsonl \
  --compare-name weighted_sft \
  --output-file outputs/generation_metrics_100_strict.md \
  --json-file outputs/generation_metrics_100_strict.json
```

## 写法模板

```text
- 自动化评测与量化突破：基于 qa_test 构建可复现评测闭环，对 baseline 与 verifier-weighted QLoRA 模型进行双重对齐评测。在 N 条测试样本上，Exact Match 由 X.XX% 提升至 Y.YY%；Reference Containment 由 A.AA% 提升至 B.BB%，增强了短答案医学问答与复杂病例分析的可验证可靠性。
```

如果后续补充 MedQA、MedMCQA 或真实业务病例集，只需要把 `--benchmark-name` 改成对应名称，并用同一脚本生成“由 X 提升至 Y”的量化表述。
