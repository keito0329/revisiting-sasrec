#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute best dropout by averaging a metric across seeds."
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name under val_by_time (e.g., Beauty, Video).",
    )
    p.add_argument(
        "--root",
        default="data/results/global_timesplit/val_by_time",
        help="Root directory containing datasets.",
    )
    p.add_argument(
        "--subdir",
        default="q09/SASRecAnalyze/test_last",
        help="Subdirectory under the dataset that contains result CSVs.",
    )
    p.add_argument(
        "--metric",
        default="test_last_NDCG@10",
        help="Metric name to average (must match metric_name in CSV).",
    )
    p.add_argument(
        "--min-dropout",
        type=float,
        default=0.1,
        help="Minimum dropout to consider.",
    )
    p.add_argument(
        "--max-dropout",
        type=float,
        default=0.5,
        help="Maximum dropout to consider.",
    )
    return p.parse_args()


def extract_metric_value(csv_path: Path, metric: str) -> float:
    df = pd.read_csv(csv_path)
    row = df.loc[df["metric_name"] == metric, "metric_value"]
    if row.empty:
        raise ValueError(f"{csv_path}: metric '{metric}' not found")
    return float(row.iloc[0])


def parse_dropout_seed_from_stem(stem: str):
    parts = stem.split("_")
    if len(parts) != 9 or parts[0] != "X":
        return None
    try:
        int(parts[1])
        int(parts[2])
        int(parts[3])
        dropout = float(parts[4])
        int(parts[5])
        float(parts[6])
        int(parts[7])
        seed = int(parts[8])
    except ValueError:
        return None
    if seed < 1 or seed > 5:
        return None
    return dropout, seed


def main():
    args = parse_args()
    base_dir = Path(args.root) / args.dataset / args.subdir
    use_no_filter_seen = False
    if not base_dir.exists():
        use_no_filter_seen = True
        base_dir = Path(args.root) / args.dataset
        if not base_dir.exists():
            raise SystemExit(f"Directory not found: {base_dir}")

    rows = []
    csv_iter = base_dir.rglob("*.csv") if use_no_filter_seen else base_dir.glob("*.csv")
    for csv_path in sorted(csv_iter):
        if use_no_filter_seen:
            if "no_filter_seen" not in csv_path.parts:
                continue
            if "test_last" not in csv_path.parts:
                continue
            parsed = parse_dropout_seed_from_stem(csv_path.stem)
        else:
            parsed = parse_dropout_seed_from_stem(csv_path.stem)
        if parsed is None:
            continue
        dropout, seed = parsed
        if not (args.min_dropout <= dropout <= args.max_dropout):
            continue
        metric_value = extract_metric_value(csv_path, args.metric)
        rows.append(
            {
                "dropout": dropout,
                "seed": seed,
                "metric": args.metric,
                "value": metric_value,
                "file": csv_path.name,
            }
        )

    if not rows:
        raise SystemExit("No matching CSV files found for the given criteria.")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("dropout", as_index=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = summary.rename(columns={"mean": "mean_value", "std": "std_value"})
    summary = summary.sort_values("mean_value", ascending=False).reset_index(drop=True)

    best = summary.iloc[0]
    print("Metric:", args.metric)
    print("Directory:", base_dir)
    if use_no_filter_seen:
        print("Mode: no_filter_seen fallback (directory missing)")
    print("")
    print("Summary (sorted by mean desc):")
    print(summary.to_string(index=False))
    print("")
    print(
        f"Best dropout: {best['dropout']} (mean={best['mean_value']:.6f}, "
        f"std={best['std_value']:.6f}, n={int(best['count'])})"
    )


if __name__ == "__main__":
    main()
