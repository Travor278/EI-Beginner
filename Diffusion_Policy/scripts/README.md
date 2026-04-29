# Scripts

This directory contains early staged scripts for scoring Push-T difficulty and
running predictive-hardness analysis.

## Current Status

These scripts are preserved for reproducibility and archaeology. They are not the
recommended entry point for new runs. Start from the final project summary and the
method-specific folders instead:

- [`../pred_hardness/`](../pred_hardness/)
- [`../dlos_dp/`](../dlos_dp/)
- [`../sfc_dp/`](../sfc_dp/)

## Main Flow

| Script | Purpose |
| --- | --- |
| `stage1_train_state_forward_probe.py` | Train a small state-forward probe |
| `stage2_score_predictive_hardness.py` | Score demos with predictive hardness |
| `stage3_make_predictive_hardness_runs.py` | Generate early comparison runs |
| `stage4_summarize_predictive_hardness_results.py` | Summarize early result logs |
| `launch_predictive_hardness_full_grid.sh` | Historical full-grid launcher |

