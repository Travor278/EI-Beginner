#!/usr/bin/env bash
set -euo pipefail

source /home/Travor/tools/miniconda3/etc/profile.d/conda.sh
conda activate robodiff-gpu
cd /home/Travor/workspaces/diffusion_policy

COMMON_ARGS=(
  --config-dir=.
  --config-name=image_pusht_diffusion_policy_cnn.yaml
  training.device=cuda:0
  training.seed=43
  training.resume=false
  logging.mode=offline
  dataloader.batch_size=8
  val_dataloader.batch_size=8
  dataloader.num_workers=0
  val_dataloader.num_workers=0
  task.dataset.max_train_episodes=90
  +task.dataset.curriculum_metric=difficulty_score_mean
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv
  task.env_runner.n_train=2
  task.env_runner.n_train_vis=1
  task.env_runner.n_test=10
  task.env_runner.n_test_vis=1
  task.env_runner.n_envs=2
  policy.down_dims=[128,256,512]
  training.num_epochs=50
  training.rollout_every=10
  training.checkpoint_every=10
  training.val_every=5
  training.sample_every=10
)

echo "===== easy_to_hard seed=43 ====="
python train.py \
  "${COMMON_ARGS[@]}" \
  +task.dataset.curriculum_mode=easy_to_hard \
  hydra.run.dir=data/outputs/pusht_easy_to_hard_seed43

echo "===== hard_to_easy seed=43 ====="
python train.py \
  "${COMMON_ARGS[@]}" \
  +task.dataset.curriculum_mode=hard_to_easy \
  hydra.run.dir=data/outputs/pusht_hard_to_easy_seed43

echo "===== all done ====="
