# Artifacts

This directory keeps lightweight result summaries and logs from the Diffusion
Policy reproduction and follow-up experiments. Heavy checkpoints, datasets, and
videos are intentionally not treated as the main record here.

## Important Files

| Path | Meaning |
| --- | --- |
| `predictive_hardness_result_summary.csv` | Early Push-T difficulty and predictive-hardness comparison summary |
| `pusht_episode_difficulty.csv` | Hand-crafted Push-T episode difficulty scores |
| `pusht_predictive_hardness_state.csv` | State-forward predictive-hardness scores |
| `state_forward_probe/` | Small state-forward probe artifacts |
| `sfc_runs_v36/` | SFC-DP A/D first-pass runs |
| `sfc_runs_diag_v36/` | SFC-DP B/D/E diagnostic runs |
| `sfc_runs_200ep/` | Long Push-T A/E confirmation runs |
| `sfc_robomimic/` | Robomimic Lift gate artifacts |

Some CSV/log files contain historical absolute WSL paths. Treat them as provenance
records, not portable commands.

