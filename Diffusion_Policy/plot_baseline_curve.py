from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_rows(log_path: Path) -> list[dict]:
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_epoch_train_loss(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([r for r in rows if "train_loss" in r and "epoch" in r])
    summary = (
        frame.groupby("epoch", as_index=False)["train_loss"]
        .mean()
        .rename(columns={"train_loss": "mean_train_loss"})
    )
    return summary


def build_eval_metrics(rows: list[dict]) -> pd.DataFrame:
    metrics = [r for r in rows if "test/mean_score" in r]
    frame = pd.DataFrame(metrics)
    return frame[["epoch", "test/mean_score", "val_loss", "train/mean_score"]].copy()


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 200,
            "savefig.dpi": 300,
        }
    )


def plot_curves(
    train_loss_by_epoch: pd.DataFrame,
    eval_metrics: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    apply_ieee_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(
        train_loss_by_epoch["epoch"],
        train_loss_by_epoch["mean_train_loss"],
        color="black",
        linewidth=1.4,
        label="Train loss",
    )
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Mean train loss")
    ax0.set_title("(a) Optimization curve")
    ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = axes[1]
    ax1.plot(
        eval_metrics["epoch"],
        eval_metrics["test/mean_score"],
        marker="o",
        markersize=4,
        color="#004488",
        linewidth=1.4,
        label="Test mean score",
    )
    ax1.plot(
        eval_metrics["epoch"],
        eval_metrics["train/mean_score"],
        marker="s",
        markersize=3.5,
        color="#BB5566",
        linewidth=1.1,
        linestyle="--",
        label="Train mean score",
    )
    best_idx = eval_metrics["test/mean_score"].idxmax()
    best_row = eval_metrics.loc[best_idx]
    ax1.scatter(
        [best_row["epoch"]],
        [best_row["test/mean_score"]],
        color="#DDAA33",
        edgecolor="black",
        linewidth=0.5,
        s=28,
        zorder=3,
    )
    ax1.annotate(
        f"Best={best_row['test/mean_score']:.3f}\n@ epoch {int(best_row['epoch'])}",
        xy=(best_row["epoch"], best_row["test/mean_score"]),
        xytext=(6, -22),
        textcoords="offset points",
        fontsize=7,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean score")
    ax1.set_ylim(0.0, 1.02)
    ax1.set_title("(b) Baseline performance")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, loc="lower right")

    fig.suptitle("Push-T baseline (Diffusion Policy, seed=42)", y=1.03, fontsize=9)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IEEE-style baseline curves.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/home/Travor/workspaces/diffusion_policy/data/outputs/pusht_baseline_seed42/logs.json.txt"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/figures"),
    )
    args = parser.parse_args()

    rows = load_rows(args.log)
    train_loss_by_epoch = build_epoch_train_loss(rows)
    eval_metrics = build_eval_metrics(rows)

    output_png = args.outdir / "pusht_baseline_seed42_ieee.png"
    output_pdf = args.outdir / "pusht_baseline_seed42_ieee.pdf"
    plot_curves(train_loss_by_epoch, eval_metrics, output_png, output_pdf)

    best_idx = eval_metrics["test/mean_score"].idxmax()
    best_row = eval_metrics.loc[best_idx]
    print(f"saved_png={output_png}")
    print(f"saved_pdf={output_pdf}")
    print(f"best_test_mean_score={best_row['test/mean_score']:.6f}")
    print(f"best_epoch={int(best_row['epoch'])}")


if __name__ == "__main__":
    main()
