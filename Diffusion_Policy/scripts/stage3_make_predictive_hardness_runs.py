from __future__ import annotations

import argparse
from pathlib import Path


def build_command(
    seed: int,
    max_train_episodes: int,
    hydra_dir: str,
    curriculum_mode: str | None,
    curriculum_metric: str | None,
    curriculum_csv_path: str | None,
) -> str:
    args = [
        "python train.py",
        "--config-dir=.",
        "--config-name=image_pusht_diffusion_policy_cnn.yaml",
        "training.device=cuda:0",
        f"training.seed={seed}",
        "training.resume=false",
        "logging.mode=offline",
        "dataloader.batch_size=8",
        "val_dataloader.batch_size=8",
        "dataloader.num_workers=0",
        "val_dataloader.num_workers=0",
        f"task.dataset.max_train_episodes={max_train_episodes}",
        "task.env_runner.n_train=2",
        "task.env_runner.n_train_vis=1",
        "task.env_runner.n_test=10",
        "task.env_runner.n_test_vis=1",
        "task.env_runner.n_envs=2",
        "policy.down_dims='[128,256,512]'",
        "training.num_epochs=50",
        "training.rollout_every=10",
        "training.checkpoint_every=10",
        "training.val_every=5",
        "training.sample_every=10",
    ]
    if curriculum_mode is not None:
        args.append(f"+task.dataset.curriculum_mode={curriculum_mode}")
    if curriculum_metric is not None:
        args.append(f"+task.dataset.curriculum_metric={curriculum_metric}")
    if curriculum_csv_path is not None:
        args.append(f"+task.dataset.curriculum_csv_path={curriculum_csv_path}")
    args.append(f"hydra.run.dir={hydra_dir}")
    return " \\\n  ".join(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Push-T experiment commands for predictive hardness runs.")
    parser.add_argument(
        "--dp-root",
        type=Path,
        default=Path("/home/Travor/workspaces/diffusion_policy"),
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_predictive_hardness_state.csv",
    )
    parser.add_argument(
        "--out-script",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/run_predictive_hardness_grid.sh"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--demo-counts", type=int, nargs="+", default=[90, 50, 20])
    args = parser.parse_args()

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {args.dp_root}",
        "",
    ]

    for demo_count in args.demo_counts:
        for seed in args.seeds:
            lines.append(f"# random | demos={demo_count} | seed={seed}")
            lines.append(
                build_command(
                    seed=seed,
                    max_train_episodes=demo_count,
                    hydra_dir=f"data/outputs/pusht_random_d{demo_count}_seed{seed}",
                    curriculum_mode=None,
                    curriculum_metric=None,
                    curriculum_csv_path=None,
                )
            )
            lines.append("")

            lines.append(f"# hand-crafted easy_to_hard | demos={demo_count} | seed={seed}")
            lines.append(
                build_command(
                    seed=seed,
                    max_train_episodes=demo_count,
                    hydra_dir=f"data/outputs/pusht_hand_easy_d{demo_count}_seed{seed}",
                    curriculum_mode="easy_to_hard",
                    curriculum_metric="difficulty_score_mean",
                    curriculum_csv_path="/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/pusht_episode_difficulty.csv",
                )
            )
            lines.append("")

            lines.append(f"# predictive hardness easy_to_hard | demos={demo_count} | seed={seed}")
            lines.append(
                build_command(
                    seed=seed,
                    max_train_episodes=demo_count,
                    hydra_dir=f"data/outputs/pusht_predictive_state_d{demo_count}_seed{seed}",
                    curriculum_mode="easy_to_hard",
                    curriculum_metric="predictive_hardness_score",
                    curriculum_csv_path=args.csv_path,
                )
            )
            lines.append("")

    args.out_script.parent.mkdir(parents=True, exist_ok=True)
    args.out_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_script={args.out_script}")


if __name__ == "__main__":
    main()

