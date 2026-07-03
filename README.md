# MedReason-SFT

First-stage data pipeline for medical SFT and process-supervision data construction.

This project uses public medical reasoning and verifiable QA datasets, keeps a traceable
source manifest, and produces cleaned intermediate JSONL files for later SFT/PRM stages.



## Stage 1

Use the Hugging Face mirror requested for this project:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\hf.exe download FreedomIntelligence/medical-o1-reasoning-SFT --repo-type dataset --local-dir data\raw\medical-o1-reasoning-SFT
.\.venv\Scripts\hf.exe download FreedomIntelligence/medical-o1-verifiable-problem --repo-type dataset --local-dir data\raw\medical-o1-verifiable-problem
.\.venv\Scripts\python.exe scripts\prepare_stage1.py --self-test
.\.venv\Scripts\python.exe scripts\prepare_stage1.py
```

Stage 1 outputs:

- `data/interim/unified_raw.jsonl`
- `data/interim/cleaned.jsonl`
- `data/interim/deduped.jsonl`
- `outputs/data_report.md`

## Stage 2

Build routed chat-format SFT splits, short-answer QA splits, holdout eval data,
analysis-pool data, and stricter seed process-supervision data:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage2.py --self-test
.\.venv\Scripts\python.exe scripts\build_stage2.py
```

Stage 2 outputs:

- `data/processed/sft_train.jsonl`
- `data/processed/sft_valid.jsonl`
- `data/processed/sft_test.jsonl`
- `data/processed/qa_train.jsonl`
- `data/processed/qa_valid.jsonl`
- `data/processed/qa_test.jsonl`
- `data/processed/process_supervision.jsonl`
- `data/processed/holdout_eval.jsonl`
- `data/processed/analysis_pool.jsonl`
- `outputs/stage2_report.md`

Long SFT examples include verifier-style metadata (`verifier_scores` and
`train_weight`) derived from answer consistency, medical safety rules,
process-supervision signal, and base quality heuristics. Low-weight long-answer
examples are routed to `analysis_pool.jsonl` instead of SFT.

## Stage 3-5 on AutoDL

Full run instructions are in [AutoDL stage 3-5 runbook](docs/autodl_stage3_5_runbook_zh.md).

Install the AutoDL training dependencies:

```bash
python -m pip install -r requirements-autodl.txt
```

Run verifier-weighted SFT with LoRA/QLoRA. The trainer applies the sample
metadata weight to assistant-token loss:

```bash
python scripts/train_weighted_sft.py \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --method qlora \
  --max-length 2048 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --output-dir outputs/stage3_qwen25_7b_weighted_sft
```

Evaluate short-answer QA separately so `qa_train.jsonl` does not change the
long-answer SFT style:

```bash
python scripts/evaluate_qa.py \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path outputs/stage3_qwen25_7b_weighted_sft/final \
  --input-file data/processed/qa_test.jsonl \
  --load-in-4bit
```

Analyze process-supervision data before any PRM training:

```bash
python scripts/analyze_process_supervision.py
```

Generate final before/after quantitative reporting after baseline and fine-tuned
QA evaluation:

```bash
python scripts/summarize_final_metrics.py
```

Plot training and QA comparison figures for tuning:

```bash
python scripts/plot_experiment_metrics.py
```

Compute HuatuoGPT-style generation metrics from prediction files:

```bash
python scripts/evaluate_generation_metrics.py \
  --input-file outputs/qa_eval_baseline_predictions_100_strict.jsonl \
  --input-name baseline \
  --compare-file outputs/qa_eval_predictions_100_strict.jsonl \
  --compare-name weighted_sft \
  --output-file outputs/generation_metrics_100_strict.md
```
