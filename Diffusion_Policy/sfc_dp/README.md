# SFC-DP Prototype

This folder contains the v3.6 SFC-DP frequency-consistency prototype.

## Status

Archived as inconclusive. The method showed promising early signals on Push-T,
but later gates did not support a robust positive claim:

- At 100 epochs, frequency supervision variants had positive signals over A.
- At 200 epochs, Push-T training saturated and the baseline caught up.
- Robomimic Lift in the tested setting saturated, with both A and E reaching
  `test/mean_score=1.00`.

Final interpretation:

- Frequency consistency may be a useful training signal.
- The current evidence is not strong enough to claim improvement over Diffusion
  Policy.
- Future testing needs a harder benchmark or lower-data regime.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | SFC-DP configuration |
| `sfc_loss.py` | Frequency-consistency loss |
| `train_sfc.py` | Push-T SFC-DP training script |
| `train_sfc_robomimic.py` | Robomimic gate training script |
| `stage1_generate_cmds.py` | Historical command generator |

See [`../Paper/研究方向v3.6_SFC-DP与后续三步计划.md`](../Paper/研究方向v3.6_SFC-DP与后续三步计划.md)
for the full experiment log.

