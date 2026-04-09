from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def build_episode_table(zarr_path: Path) -> pd.DataFrame:
    root = zarr.open(str(zarr_path), mode="r")
    action = root["data"]["action"][:]
    state = root["data"]["state"][:]
    episode_ends = root["meta"]["episode_ends"][:]

    rows = []
    start = 0
    for episode_id, end in enumerate(episode_ends):
        end = int(end)
        act = action[start:end]
        sta = state[start:end]

        num_steps = int(end - start)
        action_var = float(np.var(act, axis=0).mean())
        action_delta_mean = float(np.linalg.norm(np.diff(act, axis=0), axis=1).mean()) if len(act) > 1 else 0.0
        action_path_length = float(np.linalg.norm(np.diff(act, axis=0), axis=1).sum()) if len(act) > 1 else 0.0
        state_path_length = float(np.linalg.norm(np.diff(sta[:, :2], axis=0), axis=1).sum()) if len(sta) > 1 else 0.0

        rows.append(
            {
                "episode_id": episode_id,
                "start_idx": start,
                "end_idx": end,
                "num_steps": num_steps,
                "action_variance": action_var,
                "action_delta_mean": action_delta_mean,
                "action_path_length": action_path_length,
                "state_path_length": state_path_length,
            }
        )
        start = end

    df = pd.DataFrame(rows)
    df["length_score"] = minmax(df["num_steps"])
    df["action_var_score"] = minmax(df["action_variance"])
    df["difficulty_score_mean"] = 0.5 * (df["length_score"] + df["action_var_score"])
    df["rank_length_easy_to_hard"] = df["num_steps"].rank(method="first", ascending=True).astype(int)
    df["rank_action_var_easy_to_hard"] = df["action_variance"].rank(method="first", ascending=True).astype(int)
    df["rank_mean_easy_to_hard"] = df["difficulty_score_mean"].rank(method="first", ascending=True).astype(int)
    return df.sort_values("episode_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Push-T demonstrations for curriculum experiments.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts"),
    )
    args = parser.parse_args()

    df = build_episode_table(args.zarr_path)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "pusht_episode_difficulty.csv"
    df.to_csv(csv_path, index=False)

    easiest = df.nsmallest(5, "difficulty_score_mean")[["episode_id", "num_steps", "action_variance", "difficulty_score_mean"]]
    hardest = df.nlargest(5, "difficulty_score_mean")[["episode_id", "num_steps", "action_variance", "difficulty_score_mean"]]

    print(f"saved_csv={csv_path}")
    print(f"num_episodes={len(df)}")
    print("easiest_5")
    print(easiest.to_string(index=False))
    print("hardest_5")
    print(hardest.to_string(index=False))


if __name__ == "__main__":
    main()
