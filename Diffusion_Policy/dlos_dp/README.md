# DLOS-DP Prototype

This folder contains the v3.5 Denoising-Level Outcome Supervision prototype.

## Status

Archived after the Stage 0 world-model gate failed. The tested visual latent
world model did not outperform the trivial copy baseline:

| Metric | Value |
| --- | ---: |
| Copy baseline MSE | `0.0931` |
| OutcomeWorldModel best validation MSE | `0.2020` |

Final interpretation:

- The idea was not disproved in general.
- The chosen frozen DINO latent target was not action-sensitive enough for the
  tested Push-T setting.
- Future work should use state, object-centric, residual, or contact-aware targets
  before retrying this direction.

See [`../Paper/研究方向v3.5_去噪时世界模型一致性.md`](../Paper/研究方向v3.5_去噪时世界模型一致性.md)
for the full experiment log.

