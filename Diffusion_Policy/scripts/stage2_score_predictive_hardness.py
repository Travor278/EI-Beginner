from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predictive_hardness_common import (
    ForwardDynamicsMLP,
    build_episode_spans,
    load_pusht_arrays,
    one_step_predict,
    resolve_device,
)


def minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((series - lo) / (hi - lo)).astype(np.float32)


def rollout_error_h(
    model: ForwardDynamicsMLP,
    state: np.ndarray,
    action: np.ndarray,
    start_idx: int,
    horizon: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> float:
    pred_state = state[start_idx].copy()
    for step in range(horizon):
        pred_state = one_step_predict(
            model=model,
            state_t=pred_state,
            action_t=action[start_idx + step],
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )
    target_state = state[start_idx + horizon]
    return float(np.linalg.norm(pred_state - target_state))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Push-T episodes with predictive hardness.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/state_forward_probe/best_state_forward_probe.pt"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts"),
    )
    parser.add_argument("--rollout-horizon", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    # Stage 1 checkpoints store numpy arrays for normalization stats, so
    # PyTorch 2.6+ needs an explicit opt-out from weights_only loading.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ForwardDynamicsMLP(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    state_mean = checkpoint["state_mean"]
    state_std = checkpoint["state_std"]
    action_mean = checkpoint["action_mean"]
    action_std = checkpoint["action_std"]
    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]

    state, action, episode_ends = load_pusht_arrays(args.zarr_path)
    spans = build_episode_spans(episode_ends)

    rows: list[dict[str, float | int]] = []
    for span in spans:
        one_step_l2: list[float] = []
        one_step_mse: list[float] = []
        for idx in range(span.start_idx, span.end_idx - 1):
            pred_state = one_step_predict(
                model=model,
                state_t=state[idx],
                action_t=action[idx],
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
                target_mean=target_mean,
                target_std=target_std,
                device=device,
            )
            err = pred_state - state[idx + 1]
            one_step_l2.append(float(np.linalg.norm(err)))
            one_step_mse.append(float(np.mean(err**2)))

        rollout_l2: list[float] = []
        max_start = span.end_idx - args.rollout_horizon - 1
        if max_start >= span.start_idx:
            for idx in range(span.start_idx, max_start + 1):
                rollout_l2.append(
                    rollout_error_h(
                        model=model,
                        state=state,
                        action=action,
                        start_idx=idx,
                        horizon=args.rollout_horizon,
                        state_mean=state_mean,
                        state_std=state_std,
                        action_mean=action_mean,
                        action_std=action_std,
                        target_mean=target_mean,
                        target_std=target_std,
                        device=device,
                    )
                )

        rows.append(
            {
                "episode_id": span.episode_id,
                "start_idx": span.start_idx,
                "end_idx": span.end_idx,
                "num_steps": span.num_steps,
                "mean_one_step_l2": float(np.mean(one_step_l2)) if one_step_l2 else 0.0,
                "mean_one_step_mse": float(np.mean(one_step_mse)) if one_step_mse else 0.0,
                f"mean_rollout_l2_h{args.rollout_horizon}": float(np.mean(rollout_l2)) if rollout_l2 else 0.0,
            }
        )

    df = pd.DataFrame(rows).sort_values("episode_id").reset_index(drop=True)
    rollout_col = f"mean_rollout_l2_h{args.rollout_horizon}"
    df["one_step_score"] = minmax(df["mean_one_step_l2"])
    df["rollout_score"] = minmax(df[rollout_col])
    df["predictive_hardness_score"] = 0.5 * (df["one_step_score"] + df["rollout_score"])
    df["rank_predictive_easy_to_hard"] = df["predictive_hardness_score"].rank(method="first", ascending=True).astype(int)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "pusht_predictive_hardness_state.csv"
    df.to_csv(csv_path, index=False)

    print(f"saved_csv={csv_path}")
    print(f"num_episodes={len(df)}")
    print("easiest_5")
    print(df.nsmallest(5, "predictive_hardness_score")[["episode_id", "predictive_hardness_score", "mean_one_step_l2", rollout_col]].to_string(index=False))
    print("hardest_5")
    print(df.nlargest(5, "predictive_hardness_score")[["episode_id", "predictive_hardness_score", "mean_one_step_l2", rollout_col]].to_string(index=False))


if __name__ == "__main__":
    main()
