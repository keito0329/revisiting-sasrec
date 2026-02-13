#!/usr/bin/env python3
"""
Plot swap_scaling_recovered.csv for a single dataset.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Path to swap_scaling_recovered.csv",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K value to plot.",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset value to plot.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output figure path. Default adds _DEIM suffix.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = {"residual_scale", "k", "offset", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"{args.input} missing columns: {required - set(df.columns)}")

    sub = df[(df["k"] == args.k) & (df["offset"] == args.offset)].copy()
    if sub.empty:
        raise ValueError(f"No rows for k={args.k}, offset={args.offset} in {args.input}")
    sub = sub.sort_values("residual_scale")

    if args.output is None:
        args.output = f"swap_scaling_recovered_k{args.k}_offset{args.offset}_DEIM.pdf"
    elif not args.output.endswith("_DEIM.pdf"):
        root, ext = os.path.splitext(args.output)
        args.output = f"{root}_DEIM{ext or '.pdf'}"

    fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    ax.plot(sub["residual_scale"], sub["value"], marker="o", linewidth=1.8)
    ax.set_xlabel(r"残差スケーリング $\alpha$")
    ax.set_ylabel("回収率")
    ax.grid(True, alpha=0.25)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
