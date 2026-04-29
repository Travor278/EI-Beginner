# Diffusion Policy Research Notes

> Current status: closed archive. These notes document a completed baseline
> reproduction and several follow-up ideas that ended as negative or inconclusive
> results under the available compute and benchmark settings.

This folder is organized in the style of a research repository log: final verdict
first, then reproduction evidence, then method attempts, then raw historical notes.

## Read First

| File | Role | Current status |
| --- | --- | --- |
| [`00_项目收束总结.md`](./00_项目收束总结.md) | Final project verdict | Current |
| [`00_复现计划与环境记录.md`](./00_复现计划与环境记录.md) | Original environment and reproduction plan | Historical but useful |
| [`排错日记.md`](./排错日记.md) | Step-by-step debugging diary | Evidence for reproduction |
| [`研究方向v3.6_SFC-DP与后续三步计划.md`](./研究方向v3.6_SFC-DP与后续三步计划.md) | Most complete late-stage method log | Final method attempt |

## Research Timeline

| Stage | File | Decision |
| --- | --- | --- |
| Initial planning | [`研究方向规划.md`](./研究方向规划.md) | Broad scan of possible DP follow-ups |
| v2 | [`研究方向规划v2.md`](./研究方向规划v2.md) | Turned toward curriculum / small-data DP |
| v2 review | [`研究方向规划v2_审查与执行计划.md`](./研究方向规划v2_审查与执行计划.md) | Narrowed execution plan after novelty/risk review |
| v3 | [`研究方向v3_世界模型交叉.md`](./研究方向v3_世界模型交叉.md) | Pivoted toward world-model supervision |
| v3.1 | [`研究方向v3.1_预测难度重加权.md`](./研究方向v3.1_预测难度重加权.md) | Predictive-hardness reweighting, archived as negative |
| v3.5 | [`研究方向v3.5_去噪时世界模型一致性.md`](./研究方向v3.5_去噪时世界模型一致性.md) | DLOS-DP, stopped at failed world-model gate |
| v3.6 review | [`研究方向v3.6_三方向新颖性审查与失败分析.md`](./研究方向v3.6_三方向新颖性审查与失败分析.md) | Consolidated failed paths and new candidates |
| v3.6 final | [`研究方向v3.6_SFC-DP与后续三步计划.md`](./研究方向v3.6_SFC-DP与后续三步计划.md) | SFC-DP, inconclusive after Push-T and Lift saturation |

## Evidence Map

| Evidence type | Location |
| --- | --- |
| Baseline figure | [`../figures/pusht_baseline_seed42_ieee.png`](../figures/pusht_baseline_seed42_ieee.png) |
| Main experiment artifacts | [`../artifacts/`](../artifacts/) |
| Analysis-only artifacts | [`../artifacts_analysis_only/`](../artifacts_analysis_only/) |
| Predictive-hardness prototype | [`../pred_hardness/`](../pred_hardness/) |
| DLOS prototype | [`../dlos_dp/`](../dlos_dp/) |
| SFC-DP prototype | [`../sfc_dp/`](../sfc_dp/) |

## Current Interpretation

The correct final interpretation is conservative:

- Count the work as a completed Diffusion Policy baseline reproduction.
- Preserve follow-up ideas as useful research process and negative results.
- Do not claim a new method improvement over Diffusion Policy.
- Resume only after changing to a more discriminative benchmark, more reliable
  evaluation protocol, or larger compute budget.
