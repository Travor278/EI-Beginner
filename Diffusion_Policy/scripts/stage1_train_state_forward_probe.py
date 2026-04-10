from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from predictive_hardness_common import (
    ForwardDynamicsMLP,
    apply_standardizer,
    build_episode_spans,
    build_transition_arrays,
    fit_standardizer,
    load_pusht_arrays,
    resolve_device,
    set_seed,
    split_episode_ids,
)


def evaluate(model: ForwardDynamicsMLP, loader: DataLoader, device: torch.device) -> float:
    criterion = torch.nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for state_batch, action_batch, target_batch in loader:
            pred = model(state_batch.to(device), action_batch.to(device))
            loss = criterion(pred, target_batch.to(device))
            batch_size = state_batch.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a state-space forward probe on Push-T transitions.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/state_forward_probe"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    state, action, episode_ends = load_pusht_arrays(args.zarr_path)
    spans = build_episode_spans(episode_ends)
    train_ids, val_ids = split_episode_ids(len(spans), args.val_ratio, args.seed)

    x_train, y_train = build_transition_arrays(state, action, spans, set(train_ids))
    x_val, y_val = build_transition_arrays(state, action, spans, set(val_ids))

    state_dim = state.shape[1]
    action_dim = action.shape[1]
    state_mean, state_std = fit_standardizer(x_train[:, :state_dim])
    action_mean, action_std = fit_standardizer(x_train[:, state_dim:])
    target_mean, target_std = fit_standardizer(y_train)

    train_state = apply_standardizer(x_train[:, :state_dim], state_mean, state_std)
    train_action = apply_standardizer(x_train[:, state_dim:], action_mean, action_std)
    train_target = apply_standardizer(y_train, target_mean, target_std)

    val_state = apply_standardizer(x_val[:, :state_dim], state_mean, state_std)
    val_action = apply_standardizer(x_val[:, state_dim:], action_mean, action_std)
    val_target = apply_standardizer(y_val, target_mean, target_std)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_state),
            torch.from_numpy(train_action),
            torch.from_numpy(train_target),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_state),
            torch.from_numpy(val_action),
            torch.from_numpy(val_target),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ForwardDynamicsMLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_path = args.outdir / "best_state_forward_probe.pt"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_count = 0
        for state_batch, action_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(state_batch.to(device), action_batch.to(device))
            loss = criterion(pred, target_batch.to(device))
            loss.backward()
            optimizer.step()

            batch_size = state_batch.size(0)
            running_loss += float(loss.item()) * batch_size
            total_count += batch_size

        train_loss = running_loss / max(total_count, 1)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "hidden_dim": args.hidden_dim,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "train_episode_ids": train_ids,
                    "val_episode_ids": val_ids,
                    "seed": args.seed,
                },
                best_path,
            )

        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    history_path = args.outdir / "state_forward_probe_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary_path = args.outdir / "state_forward_probe_summary.json"
    summary = {
        "zarr_path": str(args.zarr_path),
        "best_checkpoint": str(best_path),
        "history_csv": str(history_path),
        "best_val_mse": best_val,
        "epochs": args.epochs,
        "seed": args.seed,
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "device": str(device),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved_checkpoint={best_path}")
    print(f"saved_history={history_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()

