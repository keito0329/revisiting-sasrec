import glob
import re
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent


def seed_from_path(path: Path):
    """Extract trailing integer before .csv, return None if not found."""
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def load_metrics_long_format(path: Path):
    """Files with columns [metric_name, metric_value]."""
    df = pd.read_csv(path)
    return df.set_index("metric_name")["metric_value"]


def load_metrics_wide_format(path: Path):
    """Files with first column grid_point and metrics as remaining columns."""
    df = pd.read_csv(path)
    df = df.set_index("grid_point")
    return df


def compute_effect_sizes(x: pd.Series, y: pd.Series):
    """Return Cohen's d and Hedges' g for two samples."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    diff = x.mean() - y.mean()
    pooled_var = (x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2
    if pooled_var <= 0:
        return float("nan"), float("nan")
    d = diff / (pooled_var ** 0.5)
    # Small-sample correction for unbiased effect size.
    j = 1 - (3 / (4 * (n1 + n2) - 9))
    g = d * j
    return d, g


def bucket_summary(sas_all: pd.DataFrame, light_all: pd.DataFrame, model_a="SASRecAnalyze", model_b="SFSRec", alpha=0.05):
    metrics = sorted(set(sas_all.columns) & set(light_all.columns))
    # HR@1/NDCG@1/MRR@1 are identical, so treat them as one for aggregation
    aliases = {
        "test_HitRate@1": "rank@1",
        "test_NDCG@1": "rank@1",
        "test_MRR@1": "rank@1",
        "test_last_HitRate@1": "rank@1",
        "test_last_NDCG@1": "rank@1",
        "test_last_MRR@1": "rank@1",
    }
    seen_canonical = set()
    buckets = {
        f"sig_{model_a}": 0,
        f"sig_{model_b}": 0,
        f"ns_{model_a}": 0,
        f"ns_{model_b}": 0,
        "tie": 0,
    }
    details = []

    for m in metrics:
        canonical = aliases.get(m, m)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        sa = sas_all[m]
        la = light_all[m]
        diff = sa.mean() - la.mean()
        d, g = compute_effect_sizes(sa, la)
        if len(sa) >= 2:
            t, p = stats.ttest_rel(sa, la)
        else:
            t, p = float("nan"), float("nan")
        if diff == 0:
            buckets["tie"] += 1
            better = "tie"
        elif pd.notna(p) and p < alpha:
            if diff > 0:
                buckets[f"sig_{model_a}"] += 1
                better = model_a
            else:
                buckets[f"sig_{model_b}"] += 1
                better = model_b
        else:
            if diff > 0:
                buckets[f"ns_{model_a}"] += 1
                better = model_a
            else:
                buckets[f"ns_{model_b}"] += 1
                better = model_b
        details.append((canonical, diff, d, g, better, p))

    return buckets, sorted(details, key=lambda x: -abs(x[1]))


def summarize_target_metrics(sas_all, light_all, model_a="SASRecAnalyze", model_b="SFSRec", alpha=0.05, targets=None):
    """Return summary dict for selected metrics (e.g., HR/NDCG@1,10,100)."""
    if targets is None:
        targets = []
    filtered = [m for m in sas_all.columns if m in targets]
    summary = []
    for m in filtered:
        sa = sas_all[m]
        la = light_all[m]
        diff = sa.mean() - la.mean()
        d, g = compute_effect_sizes(sa, la)
        if len(sa) >= 2:
            t, p = stats.ttest_rel(sa, la)
        else:
            t, p = float("nan"), float("nan")
        if diff > 0:
            winner = model_a
        elif diff < 0:
            winner = model_b
        else:
            winner = "tie"
        sig = pd.notna(p) and p < alpha
        summary.append((m, sa.mean(), la.mean(), winner, sig, p, d, g))
    return summary


def analyze_leave_one_out(
    dataset="Sports",
    model_a="SASRecAnalyze",
    model_b="SFSRec",
    dropout="0.5",
    seq_len="50",
):
    def _loo_glob(model: str, dropout: str, seq_len: str):
        if model == "SASRecAnalyze":
            pattern = f"X_64_2_2_{dropout}_{seq_len}_1.0*.csv"
        elif model == "SFSRec":
            pattern = f"X_64_2_{dropout}_{seq_len}*.csv"
        else:
            pattern = f"X_64_2_2_{dropout}_{seq_len}_*.csv"
        return [Path(p) for p in glob.glob(str(ROOT / f"data/results/leave-one-out/{dataset}/{model}/test/{pattern}"))]

    sas_paths = _loo_glob(model_a, dropout, seq_len)
    light_paths = _loo_glob(model_b, dropout, seq_len)

    sas_df = pd.DataFrame({"path": sas_paths})
    sas_df["seed"] = sas_df["path"].apply(seed_from_path)
    light_df = pd.DataFrame({"path": light_paths})
    light_df["seed"] = light_df["path"].apply(seed_from_path)

    sas_df = sas_df.dropna(subset=["seed"])
    light_df = light_df.dropna(subset=["seed"])
    pairs = sas_df.merge(light_df, on="seed", suffixes=("_sas", "_light")).sort_values("seed")

    sas_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_sas) for _, row in pairs.iterrows()}).T
    light_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_light) for _, row in pairs.iterrows()}).T

    buckets, details = bucket_summary(sas_all, light_all, model_a=model_a, model_b=model_b)

    print(f"=== Leave-one-out results {dataset} ({model_a} vs {model_b}) ===")
    print("seeds:", list(pairs["seed"]))
    print("metrics:", len(set(sas_all.columns) & set(light_all.columns)))
    print("bucket counts:", buckets)
    print(f"mean {model_a}:\n", sas_all.mean())
    print(f"mean {model_b}:\n", light_all.mean())
    print("top diffs:")
    for m, d, d_eff, g_eff, b, p in details[:10]:
        print(f"  {m}: diff={d:.4g} d={d_eff:.4g} g={g_eff:.4g} ({b}), p={p:.3g}")
    targets = [m for m in sas_all.columns if (m.endswith("@1") or m.endswith("@10") or m.endswith("@100")) and ("HitRate" in m or "NDCG" in m)]
    def _metric_sort_key(m: str):
        metric_priority = 0 if "HitRate" in m else 1
        k_map = {"@1": 1, "@10": 10, "@100": 100}
        k_val = next((v for k, v in k_map.items() if m.endswith(k)), 999)
        return (metric_priority, k_val, m)
    targets = sorted(targets, key=_metric_sort_key)
    summary = summarize_target_metrics(sas_all, light_all, model_a=model_a, model_b=model_b, targets=targets)
    print(f"HR/NDCG @1,@10,@100 summary (mean_{model_a}, mean_{model_b}, winner, significant, p, d, g):")
    for m, sa_mean, la_mean, winner, sig, p, d, g in summary:
        print(f"  {m}: {sa_mean:.4g} vs {la_mean:.4g} -> {winner}, sig={sig}, p={p:.3g}, d={d:.4g}, g={g:.4g}")


def analyze_global_timesplit(
    dataset="Sports",
    quantile="q09",
    file_name="test_last.csv",
    model_a="SASRecAnalyze",
    model_b="SFSRec",
    dropout="0.5",
    seq_len="50",
):
    base = ROOT / f"data/results/global_timesplit/val_by_time/{dataset}/{quantile}"

    def _test_last_glob(model: str, dropout: str, seq_len: str):
        # Prefer standard path, then fallback to no_filter_seen
        if model == "SASRecAnalyze":
            pattern = f"X_64_2_2_{dropout}_{seq_len}_1.0*.csv"
        elif model == "SFSRec":
            pattern = f"X_64_2_{dropout}_{seq_len}*.csv"
        else:
            pattern = f"X_64_2_2_{dropout}_{seq_len}_*.csv"
        paths = [
            base / model / "test_last",
            base / model / "no_filter_seen" / "test_last",
        ]
        for p in paths:
            g = list(p.glob(pattern))
            if g:
                return g, p
        return [], None

    def _final_results_path(model: str):
        # Prefer standard path, then fallback to no_filter_seen
        paths = [
            base / model / "final_results" / file_name,
            base / model / "no_filter_seen" / "final_results" / file_name,
        ]
        for p in paths:
            if p.exists():
                return p
        return None

    # 1) Try X_*.csv under test_last/ (same style as leave-one-out)
    sas_glob, sas_dir = _test_last_glob(model_a, dropout, seq_len)
    light_glob, light_dir = _test_last_glob(model_b, dropout, seq_len)

    if sas_glob and light_glob:
        sas_df = pd.DataFrame({"path": sas_glob})
        sas_df["seed"] = sas_df["path"].apply(seed_from_path)
        light_df = pd.DataFrame({"path": light_glob})
        light_df["seed"] = light_df["path"].apply(seed_from_path)
        sas_df = sas_df.dropna(subset=["seed"])
        light_df = light_df.dropna(subset=["seed"])
        pairs = sas_df.merge(light_df, on="seed", suffixes=("_sas", "_light")).sort_values("seed")

        if pairs.empty:
            print("Global timesplit X_*.csv found but no matching seeds.")
            return

        sas_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_sas) for _, row in pairs.iterrows()}).T
        light_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_light) for _, row in pairs.iterrows()}).T
        index_info = list(pairs["seed"])
    else:
        # 2) Fallback to aggregated final_results/<file_name>
        sas_path = _final_results_path(model_a)
        light_path = _final_results_path(model_b)
        if sas_path is None or light_path is None:
            print("Global timesplit files not found:", sas_path, light_path)
            return

        sas_df = load_metrics_wide_format(sas_path)
        light_df = load_metrics_wide_format(light_path)
        common_idx = sas_df.index.intersection(light_df.index)
        sas_all = sas_df.loc[common_idx]
        light_all = light_df.loc[common_idx]
        index_info = list(common_idx)

    buckets, details = bucket_summary(sas_all, light_all, model_a=model_a, model_b=model_b)
    header_note = []
    if sas_dir is not None and "no_filter_seen" in str(sas_dir):
        header_note.append(f"{model_a}:no_filter_seen")
    if light_dir is not None and "no_filter_seen" in str(light_dir):
        header_note.append(f"{model_b}:no_filter_seen")
    note = f" [{', '.join(header_note)}]" if header_note else ""
    print(f"\n=== Global timesplit val_by_time results {dataset} ({model_a} vs {model_b}){note} ===")
    print("pairs:", index_info)
    print("metrics:", len(set(sas_all.columns) & set(light_all.columns)))
    print("bucket counts:", buckets)
    print(f"mean {model_a}:\n", sas_all.mean())
    print(f"mean {model_b}:\n", light_all.mean())
    print("top diffs:")
    for m, d, d_eff, g_eff, b, p in details[:10]:
        print(f"  {m}: diff={d:.4g} d={d_eff:.4g} g={g_eff:.4g} ({b}), p={p:.3g}")
    targets = [m for m in sas_all.columns if (m.endswith("@1") or m.endswith("@10") or m.endswith("@100")) and ("HitRate" in m)]
    summary = summarize_target_metrics(sas_all, light_all, model_a=model_a, model_b=model_b, targets=targets)
    print(f"HR @1,@10,@100 summary (mean_{model_a}, mean_{model_b}, winner, significant, p, d, g):")
    for m, sa_mean, la_mean, winner, sig, p, d, g in summary:
        print(f"  {m}: {sa_mean:.4g} vs {la_mean:.4g} -> {winner}, sig={sig}, p={p:.3g}, d={d:.4g}, g={g:.4g}")

    targets = [m for m in sas_all.columns if (m.endswith("@10") or m.endswith("@100")) and ("NDCG" in m)]
    summary = summarize_target_metrics(sas_all, light_all, model_a=model_a, model_b=model_b, targets=targets)
    print(f"NDCG @10,@100 summary (mean_{model_a}, mean_{model_b}, winner, significant, p, d, g):")
    for m, sa_mean, la_mean, winner, sig, p, d, g in summary:
        print(f"  {m}: {sa_mean:.4g} vs {la_mean:.4g} -> {winner}, sig={sig}, p={p:.3g}, d={d:.4g}, g={g:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str, help="dataset name (e.g., Sports)")
    ap.add_argument("--quantile", type=str, default="q09")
    ap.add_argument("--file-name", type=str, default="X_64_2_2_0.5_50_1.0*.csv")
    ap.add_argument("--model-a", type=str, default="SASRecAnalyze")
    ap.add_argument("--model-b", type=str, default="SFSRec")
    ap.add_argument("--dropout", type=str, default="0.5")
    ap.add_argument("--seq-len", type=str, default="50")
    ap.add_argument("--skip-loo", action="store_true", help="skip leave-one-out analysis")
    args = ap.parse_args()

    if not args.skip_loo:
        analyze_leave_one_out(
            dataset=args.dataset,
            model_a=args.model_a,
            model_b=args.model_b,
            dropout=args.dropout,
            seq_len=args.seq_len,
        )
    analyze_global_timesplit(
        dataset=args.dataset,
        quantile=args.quantile,
        file_name=args.file_name,
        model_a=args.model_a,
        model_b=args.model_b,
        dropout=args.dropout,
        seq_len=args.seq_len,
    )
