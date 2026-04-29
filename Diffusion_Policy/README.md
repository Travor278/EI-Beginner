# Diffusion Policy Reproduction and Research Log

> Status: closed as a baseline reproduction plus archived negative-result research notes.
> Date: 2026-04-29.

This folder records the reproduction of the classic imitation-learning baseline
Diffusion Policy and the follow-up research attempts around curriculum learning,
predictive hardness, world-model supervision, and frequency consistency.

The result to count for the learning roadmap is:

> We completed the reproduction path for the classic imitation-learning baseline:
> Diffusion Policy, <https://diffusion-policy.cs.columbia.edu>.

The follow-up methods were implemented and tested, but they are archived as
negative or inconclusive results rather than presented as successful new methods.

## TL;DR

- Official Diffusion Policy reproduction: completed at the engineering/baseline level.
- Push-T training, rollout, validation, logging, and checkpoint paths were made runnable.
- The old official environment was not suitable for the local modern GPU stack, so a
  newer `robodiff-gpu` environment was used for real GPU runs.
- Several research extensions were explored, but none produced a robust positive
  result under the available GPU/task/evaluation setting.
- The folder is now organized as a research archive: reproducible notes, result
  summaries, code experiments, and negative-result evidence are kept together.

## What Can Be Claimed

| Claim | Status | Evidence |
| --- | --- | --- |
| Diffusion Policy baseline reproduction path was completed | Done | [`Paper/排错日记.md`](./Paper/排错日记.md), [`figures/`](./figures/) |
| Official environment compatibility was diagnosed and patched around | Done | [`Paper/00_复现计划与环境记录.md`](./Paper/00_复现计划与环境记录.md) |
| Push-T baseline and analysis artifacts were produced | Done | [`artifacts/`](./artifacts/), [`figures/pusht_baseline_seed42_ieee.png`](./figures/pusht_baseline_seed42_ieee.png) |
| Curriculum / hardness / world-model / frequency variants clearly beat DP | Not claimed | See [`Paper/00_项目收束总结.md`](./Paper/00_项目收束总结.md) |
| This folder contains paper-ready positive method results | Not claimed | Archived as negative/inconclusive research notes |

## Directory Map

```text
Diffusion_Policy/
├─ README.md                         # current entry point
├─ Paper/                            # research notes, final summary, negative results
├─ figures/                          # selected plots for the baseline record
├─ artifacts/                        # result summaries and lightweight logs
├─ artifacts_analysis_only/          # smoke tests and analysis-only artifacts
├─ artifacts_shareable*/             # shareable snapshots generated during the work
├─ official_pusht_code/              # D-drive copy of upstream Push-T/Diffusion Policy code, no data or outputs
├─ scripts/                          # early staged analysis scripts
├─ pred_hardness/                    # v3.1 predictive-hardness reweighting prototype
├─ dlos_dp/                          # v3.5 world-model / DLOS prototype
├─ sfc_dp/                           # v3.6 SFC-DP frequency-consistency prototype
├─ plot_baseline_curve.py            # baseline curve plotting helper
└─ score_pusht_difficulty.py         # early Push-T difficulty scoring helper
```

## Research Outcome Matrix

| Track | Main idea | Outcome |
| --- | --- | --- |
| Baseline | Reproduce Diffusion Policy on Push-T | Completed as the stable baseline record |
| v2 curriculum | Hand-crafted difficulty order | Negative, hand-crafted difficulty did not help |
| v3.1 PHRew | Predictive-hardness sample reweighting | Negative, `soft_hard` underperformed `uniform` |
| v3.5 DLOS | Denoising-level world-model supervision | Gate failed, visual latent WM did not beat copy baseline |
| v3.6 SFC-DP | Frequency consistency during denoising | Inconclusive, early signal existed but final gates saturated |

## Recommended Reading Order

1. [`Paper/README.md`](./Paper/README.md) for the curated research-note index.
2. [`Paper/00_项目收束总结.md`](./Paper/00_项目收束总结.md) for the final verdict.
3. [`Paper/排错日记.md`](./Paper/排错日记.md) for the reproduction/debugging trail.
4. [`Paper/研究方向v3.6_SFC-DP与后续三步计划.md`](./Paper/研究方向v3.6_SFC-DP与后续三步计划.md) for the most complete later-stage experiment log.

## Resume Policy

Do not spend more compute on the same Push-T or current Robomimic Lift setting.
The archived logs indicate that these settings are either too noisy or too saturated
to support a clean method claim. A future restart should first change the benchmark
or data regime, for example low-data Lift/Can, Robomimic Can, LIBERO, or a setting
with more reliable evaluation episodes.
