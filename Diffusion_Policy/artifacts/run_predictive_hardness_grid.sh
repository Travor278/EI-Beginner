#!/usr/bin/env bash
set -euo pipefail
cd /home/Travor/workspaces/diffusion_policy

# random | demos=90 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=90 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  hydra.run.dir=data/outputs/pusht_random_d90_seed42

# hand-crafted easy_to_hard | demos=90 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=90 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=difficulty_score_mean \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv \
  hydra.run.dir=data/outputs/pusht_hand_easy_d90_seed42

# predictive hardness easy_to_hard | demos=90 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=90 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=predictive_hardness_score \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_predictive_hardness_state.csv \
  hydra.run.dir=data/outputs/pusht_predictive_state_d90_seed42

# random | demos=90 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=90 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  hydra.run.dir=data/outputs/pusht_random_d90_seed43

# hand-crafted easy_to_hard | demos=90 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=90 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=difficulty_score_mean \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv \
  hydra.run.dir=data/outputs/pusht_hand_easy_d90_seed43

# predictive hardness easy_to_hard | demos=90 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=90 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=predictive_hardness_score \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_predictive_hardness_state.csv \
  hydra.run.dir=data/outputs/pusht_predictive_state_d90_seed43

# random | demos=50 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=50 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  hydra.run.dir=data/outputs/pusht_random_d50_seed42

# hand-crafted easy_to_hard | demos=50 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=50 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=difficulty_score_mean \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv \
  hydra.run.dir=data/outputs/pusht_hand_easy_d50_seed42

# predictive hardness easy_to_hard | demos=50 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=50 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=predictive_hardness_score \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_predictive_hardness_state.csv \
  hydra.run.dir=data/outputs/pusht_predictive_state_d50_seed42

# random | demos=50 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=50 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  hydra.run.dir=data/outputs/pusht_random_d50_seed43

# hand-crafted easy_to_hard | demos=50 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=50 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=difficulty_score_mean \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv \
  hydra.run.dir=data/outputs/pusht_hand_easy_d50_seed43

# predictive hardness easy_to_hard | demos=50 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=50 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=predictive_hardness_score \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_predictive_hardness_state.csv \
  hydra.run.dir=data/outputs/pusht_predictive_state_d50_seed43

# random | demos=20 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=20 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  hydra.run.dir=data/outputs/pusht_random_d20_seed42

# hand-crafted easy_to_hard | demos=20 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=20 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=difficulty_score_mean \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv \
  hydra.run.dir=data/outputs/pusht_hand_easy_d20_seed42

# predictive hardness easy_to_hard | demos=20 | seed=42
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=20 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=predictive_hardness_score \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_predictive_hardness_state.csv \
  hydra.run.dir=data/outputs/pusht_predictive_state_d20_seed42

# random | demos=20 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=20 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  hydra.run.dir=data/outputs/pusht_random_d20_seed43

# hand-crafted easy_to_hard | demos=20 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=20 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=difficulty_score_mean \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv \
  hydra.run.dir=data/outputs/pusht_hand_easy_d20_seed43

# predictive hardness easy_to_hard | demos=20 | seed=43
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.device=cuda:0 \
  training.seed=43 \
  training.resume=false \
  logging.mode=offline \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  task.dataset.max_train_episodes=20 \
  task.env_runner.n_train=2 \
  task.env_runner.n_train_vis=1 \
  task.env_runner.n_test=10 \
  task.env_runner.n_test_vis=1 \
  task.env_runner.n_envs=2 \
  policy.down_dims='[128,256,512]' \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=5 \
  training.sample_every=10 \
  +task.dataset.curriculum_mode=easy_to_hard \
  +task.dataset.curriculum_metric=predictive_hardness_score \
  +task.dataset.curriculum_csv_path=/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_predictive_hardness_state.csv \
  hydra.run.dir=data/outputs/pusht_predictive_state_d20_seed43

