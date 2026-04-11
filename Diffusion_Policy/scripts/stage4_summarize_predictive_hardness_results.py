from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_best_score(log_path: Path) -> tuple[float | None, int | None]:
    if not log_path.exists():
        return None, None
    best_score = None
    best_epoch = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "test/mean_score" not in row:
            continue
        score = float(row["test/mean_score"])
        epoch = int(row.get("epoch", -1))
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
    return best_score, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Push-T experiment results from multiple run directories.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec in the form method,seed,run_dir . Repeat this flag for multiple runs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts"),
    )
    args = parser.parse_args()

    rows: list[dict[str, str | int | float]] = []
    for spec in args.run:
        method, seed, run_dir = [item.strip() for item in spec.split(",", 2)]
        log_path = Path(run_dir) / "logs.json.txt"
        best_score, best_epoch = parse_best_score(log_path)
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "run_dir": run_dir,
                "best_test_mean_score": best_score,
                "best_epoch": best_epoch,
                "log_path": str(log_path),
            }
        )

    frame = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "predictive_hardness_result_summary.csv"
    frame.to_csv(csv_path, index=False)

    print(f"saved_csv={csv_path}")
    if not frame.empty:
        try:
            print(frame.to_markdown(index=False))
        except ImportError:
            # Keep stage4 dependency-light: tabulate is optional in pandas.
            print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
