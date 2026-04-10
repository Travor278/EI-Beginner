#!/usr/bin/env bash
set -euo pipefail

source /home/Travor/tools/miniconda3/etc/profile.d/conda.sh
conda activate robodiff-gpu

bash /mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/run_predictive_hardness_grid.sh

python /mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/scripts/stage4_summarize_predictive_hardness_results.py \
  --outdir /mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts \
  --run random_d90,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_random_d90_seed42 \
  --run hand_easy_d90,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_hand_easy_d90_seed42 \
  --run predictive_state_d90,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_predictive_state_d90_seed42 \
  --run random_d90,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_random_d90_seed43 \
  --run hand_easy_d90,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_hand_easy_d90_seed43 \
  --run predictive_state_d90,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_predictive_state_d90_seed43 \
  --run random_d50,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_random_d50_seed42 \
  --run hand_easy_d50,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_hand_easy_d50_seed42 \
  --run predictive_state_d50,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_predictive_state_d50_seed42 \
  --run random_d50,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_random_d50_seed43 \
  --run hand_easy_d50,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_hand_easy_d50_seed43 \
  --run predictive_state_d50,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_predictive_state_d50_seed43 \
  --run random_d20,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_random_d20_seed42 \
  --run hand_easy_d20,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_hand_easy_d20_seed42 \
  --run predictive_state_d20,42,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_predictive_state_d20_seed42 \
  --run random_d20,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_random_d20_seed43 \
  --run hand_easy_d20,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_hand_easy_d20_seed43 \
  --run predictive_state_d20,43,/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_predictive_state_d20_seed43
