# Predictive Hardness Reweighting

This folder contains the v3.1 predictive-hardness reweighting prototype.

## Status

Archived as a negative result. The implementation worked, but the main
`soft_hard` strategy underperformed the `uniform` baseline on both recorded seeds.

Final interpretation:

- Predictive hardness can be computed and connected to training.
- Chunk-level reweighting alone was not enough to improve control.
- This path motivated later loss-level or representation-level supervision ideas.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Shared PHRew config |
| `dataset.py` | Dataset utilities |
| `chunk_scorer.py` | Chunk hardness scoring |
| `stage2b_score_chunks.py` | Score chunks with the trained probe |
| `stage3_train_phrew.py` | Train PHRew variants |
| `stage4_generate_cmds.py` | Generate comparison commands |

See [`../Paper/研究方向v3.1_预测难度重加权.md`](../Paper/研究方向v3.1_预测难度重加权.md)
for the full reasoning and results.

