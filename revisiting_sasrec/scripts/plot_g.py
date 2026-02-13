#!/usr/bin/env python3
"""Plot mean Hedge's g against Mixing Ratio (AttnResLN)."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

# Avoid permission issues when matplotlib tries to write cache in restricted envs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


# Hedge's g per metric in order:
# [HR@1, HR@10, HR@100, NDCG@10, NDCG@100]
G_VALUES: Dict[str, List[float]] = {
    "Beauty": [0.5794, 6.26e-15, -0.2539, 0.2976, 0.003446],
    "Sports": [-1.061, -0.4941, -1.01, -1.026, -1.29],
    "Toys": [0.5962, -0.2843, -0.4459, 0.2315, 0.006269],
    "Gowalla": [-0.8936, 0.1899, 0.5741, -0.2649, -0.03959],
    "ML-20M": [0.8513, 2.311, 1.714, 1.995, 1.932],
    "ML-1M": [0.5398, 1.403, 4.121, 1.833, 3.457],
    "OTTO": [0.4549, 0.5202, -0.3872, 0.924, 0.3742],
    # "Zvuk": [-2.1, -6.287, -4.104, -5.475, -8.21],
    "Steam": [-0.6495, -1.038, -0.6151, -0.9306, -0.7362],
    "YooChoose": [-0.5795, -0.2507, 0.09791, -1.475, -2.653],
    "LastFM": [4.843e-16, -0.8414, -0.6235, -0.3957, -0.7374],
    "Diginetica": [0.07754, -0.4648, 0.2442, -1.176, -0.711],
    "BeerAdvocate" : [-0.7884, -1.129, -2.038, -1.151, -1.891], 
    "Foursquare" : [1.196, -1.336, -0.754, -1.097, -0.8739],
    "Yelp" : [-0.6233, -1.362, -1.247, -1.157, -1.204],
}

# Mixing Ratio (AttnResLN)
MIXING_RATIO: Dict[str, float] = {
    "Beauty": 0.1496,
    "Sports": 0.1460,
    "Toys": 0.1456,
    "Diginetica": 0.2359,
    "YooChoose": 0.2330,
    "ML-1M": 0.4099,
    "Steam": 0.3069,
    # "Zvuk": 0.2710,
    "OTTO": 0.1742,
    "ML-20M": 0.3409,
    "LastFM": 0.1546,
    "Gowalla": 0.3324,
    "BeerAdvocate": 0.3006,
    "Foursquare": 0.4064998,
    "Yelp": 0.292392,
}


def build_common_arrays() -> tuple[list[str], np.ndarray, np.ndarray]:
    common = [name for name in G_VALUES if name in MIXING_RATIO]
    g_means = np.array([np.mean(G_VALUES[name]) for name in common], dtype=float)
    mixing = np.array([MIXING_RATIO[name] for name in common], dtype=float)
    return common, g_means, mixing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="plots/g_mean_vs_mixing_ratio.pdf",
        help="Output figure path.",
    )
    args = ap.parse_args()

    datasets, g_mean, mixing = build_common_arrays()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    plt.rcParams.update(
        {
            "axes.titlesize": 34,
            "axes.labelsize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
        }
    )

    fig, ax = plt.subplots(figsize=(14.0, 8.8), constrained_layout=True)
    ax.scatter(mixing, g_mean, s=130, color="#4c78a8", alpha=0.9)
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)

    x_min, x_max = float(np.min(mixing)), float(np.max(mixing))
    y_min, y_max = float(np.min(g_mean)), float(np.max(g_mean))
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Keep labels near each point and avoid overlap with both labels and points.
    candidates = [
        (14, 14), (14, -14), (-14, 14), (-14, -14),
        (24, 0), (-24, 0), (0, 24), (0, -24),
        (30, 16), (30, -16), (-30, 16), (-30, -16),
        (36, 0), (-36, 0), (0, 36), (0, -36),
        (44, 18), (44, -18), (-44, 18), (-44, -18),
    ]
    forced_offsets = {
        "ML-1M": (-30, -20),      # left-down
        "Foursquare": (-34, -20), # left-down
        "LastFM": (26, -18),      # right-down
        "Beauty": (0, 30),        # up
        "Toys": (0, -34),         # down
        "Sports": (0, -30),       # down
        "ML-20M": (-32, -22),     # left-down
        "Yelp": (-28, 24),        # left-up
        "Diginetica": (0, 52),    # more up
    }
    points = sorted(zip(datasets, mixing, g_mean), key=lambda v: (v[1], v[2]))
    point_pixels = [ax.transData.transform((x, y)) for _, x, y in points]
    marker_radius_px = 13
    placed_boxes = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for name, x, y in points:
        placed = False
        if name in forced_offsets:
            dx, dy = forced_offsets[name]
            ann = ax.annotate(
                name,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=26,
                ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                va="bottom" if dy >= 0 else "top",
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.9},
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.9, "shrinkA": 4, "shrinkB": 3},
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
            placed_boxes.append(bbox)
            continue

        trial_offsets = candidates
        for dx, dy in trial_offsets:
            ann = ax.annotate(
                name,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=26,
                ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                va="bottom" if dy >= 0 else "top",
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.9},
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.9, "shrinkA": 4, "shrinkB": 3},
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
            overlaps_label = any(bbox.overlaps(prev) for prev in placed_boxes)
            overlaps_point = any(
                (bbox.x0 - marker_radius_px) <= px <= (bbox.x1 + marker_radius_px)
                and (bbox.y0 - marker_radius_px) <= py <= (bbox.y1 + marker_radius_px)
                for px, py in point_pixels
            )
            if overlaps_label or overlaps_point:
                ann.remove()
                continue
            placed_boxes.append(bbox)
            placed = True
            break
        if not placed:
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=26,
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.9},
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.9, "shrinkA": 4, "shrinkB": 3},
            )

    ax.set_xlabel("Mixing Ratio (AttnResLN)")
    ax.set_ylabel("Mean Hedge's $g$")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(x_min - 0.07 * x_span, x_max + 0.07 * x_span)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.08 * y_span)

    fig.savefig(args.output)

    missing = sorted(set(G_VALUES) - set(MIXING_RATIO))
    if missing:
        print(f"Skipped datasets without Mixing Ratio: {', '.join(missing)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
