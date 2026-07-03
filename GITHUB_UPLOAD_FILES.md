# GitHub Upload Scope

This repository package is intended to publish the reproducible code, documents,
metadata rules, and lightweight experiment figures for MedReason-SFT.

## Include

- `README.md`
- `requirements-autodl.txt`
- `.gitignore`
- `scripts/`
- `docs/autodl_stage3_5_runbook_zh.md`
- `docs/final_metrics_template_zh.md`
- `data/metadata/`
- `plots/`

## Exclude

- `.venv/`
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `outputs/`
- `docs/stage1_stage2_lessons.md`
- `docs/stage1_stage2_lessons_zh.md`
- model weights and checkpoints
- local logs and generated archive files

Large datasets are regenerated from public source datasets by running the stage
1 and stage 2 scripts. Training outputs, adapters, and checkpoints should not be
committed directly to GitHub.
