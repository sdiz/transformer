import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def _list_runs(group_dir: Path) -> List[Path]:
    if not group_dir.exists():
        return []
    return sorted([p for p in group_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])


def _is_period_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name
    return len(name) >= 4 and name[:4].isdigit()


def _read_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.json"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


CFG_KEYS = [
    "seq_length",
    "z_window",
    "use_cls_token",
    "zscore",
    "layers",
    "head_size",
    "num_heads",
    "ff_dim",
    "dropout",
    "attn_dropout",
    "calibration",
    "refit",
    "timeframe",
    "use_returns",
    "use_spx_close",
    "data_start_year",
    "end_date",
]


def _cfg_columns(cfg: Dict) -> Dict[str, object]:
    cols: Dict[str, object] = {}
    for k in CFG_KEYS:
        if k in cfg:
            cols[f"cfg_{k}"] = cfg.get(k)
    return cols


def _read_wf_metrics(run_dir: Path, group: str) -> Optional[pd.DataFrame]:
    df = _safe_read_csv(run_dir / "wf_metrics.csv")
    if df is None or df.empty:
        return None
    df = df.copy()
    # Normalize types
    if "period" in df.columns:
        df["period"] = df["period"].astype(str)
    df.insert(0, "run_id", run_dir.name)
    df.insert(0, "group", group)
    return df


def _read_year_series(year_dir: Path, run_id: str, group: str) -> Optional[pd.DataFrame]:
    year = year_dir.name
    f = year_dir / f"wf_{year}_series.csv"
    df = _safe_read_csv(f)
    if df is None or df.empty:
        return None
    df = df.copy()
    # Ensure first column is Date
    first = df.columns[0]
    if first.lower() != "date":
        df.rename(columns={first: "Date"}, inplace=True)
    # Parse Date
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception:
        pass
    # Keep known columns only if present
    keep_cols = [c for c in ["Date", "Return", "VaR", "Breach", "GARCH_VaR", "LSTM_VaR"] if c in df.columns]
    df = df[keep_cols]
    # Normalize Breach to bool
    if "Breach" in df.columns:
        if df["Breach"].dtype == object:
            df["Breach"] = df["Breach"].astype(str).str.lower().map({"true": True, "false": False})
        # If numeric 0/1, map to bool
        if pd.api.types.is_numeric_dtype(df["Breach"]):
            df["Breach"] = df["Breach"].astype(float).round().astype(int).astype(bool)
    df.insert(0, "period", year)
    df.insert(0, "run_id", run_id)
    df.insert(0, "group", group)
    return df


def _read_year_metrics_compare(year_dir: Path, run_id: str, group: str) -> Optional[pd.DataFrame]:
    year = year_dir.name
    f = year_dir / f"wf_{year}_metrics_compare.csv"
    df = _safe_read_csv(f)
    if df is None or df.empty:
        return None
    df = df.copy()
    # Ensure period is str and correct
    if "period" in df.columns:
        df["period"] = str(year)
    else:
        df.insert(0, "period", str(year))
    df.insert(0, "run_id", run_id)
    df.insert(0, "group", group)
    return df


def _read_year_permutation(year_dir: Path, run_id: str, group: str) -> Optional[pd.DataFrame]:
    year = year_dir.name
    f = year_dir / f"wf_{year}_permutation_importance.csv"
    df = _safe_read_csv(f)
    if df is None or df.empty:
        return None
    df = df.copy()
    df.insert(0, "period", str(year))
    df.insert(0, "run_id", run_id)
    df.insert(0, "group", group)
    return df


def _zscore_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Expect first column is Date-like; drop it from features
    features = [c for c in df.columns if c != df.columns[0]]
    if len(features) == 0:
        return pd.DataFrame()
    X = df[features].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame({
        "feature": X.columns,
        "mean": X.mean(skipna=True).values,
        "std": X.std(ddof=0, skipna=True).values,
        "p05": X.quantile(0.05, interpolation="linear").values,
        "p50": X.quantile(0.50, interpolation="linear").values,
        "p95": X.quantile(0.95, interpolation="linear").values,
        "num_obs": X.notna().sum().values,
        "frac_nan": X.isna().mean().values,
    })
    return summary


def _read_year_zscore_summary(year_dir: Path, run_id: str, group: str) -> Optional[pd.DataFrame]:
    year = year_dir.name
    f = year_dir / f"wf_{year}_zscore_oos.csv"
    df = _safe_read_csv(f)
    if df is None or df.empty:
        return None
    try:
        summ = _zscore_summary(df)
    except Exception:
        return None
    if summ.empty:
        return None
    summ.insert(0, "period", str(year))
    summ.insert(0, "run_id", run_id)
    summ.insert(0, "group", group)
    return summ


def aggregate_group(
    root: Path,
    group: str,
    out_dir: Path,
    include_series: bool = True,
    include_metrics: bool = True,
    include_compare: bool = True,
    include_permutation: bool = True,
    zscore_mode: str = "summary",  # none|summary
    verbose: bool = True,
):
    group_dir = root / group
    runs = _list_runs(group_dir)
    if verbose:
        print(f"Scanning {group!r} in {group_dir} – found {len(runs)} runs")
    if not runs:
        return

    coll_metrics: List[pd.DataFrame] = []
    coll_series: List[pd.DataFrame] = []
    coll_compare: List[pd.DataFrame] = []
    coll_perm: List[pd.DataFrame] = []
    coll_zsum: List[pd.DataFrame] = []

    for run_dir in runs:
        run_id = run_dir.name
        cfg = _read_config(run_dir)
        cfg_cols = _cfg_columns(cfg)
        if include_metrics:
            m = _read_wf_metrics(run_dir, group)
            if m is not None and not m.empty:
                # attach config columns
                for k, v in cfg_cols.items():
                    m[k] = v
                coll_metrics.append(m)

        # Iterate period subfolders once and read what's needed
        for year_dir in sorted([p for p in run_dir.iterdir() if _is_period_dir(p)]):
            if include_series:
                s = _read_year_series(year_dir, run_id, group)
                if s is not None and not s.empty:
                    for k, v in cfg_cols.items():
                        s[k] = v
                    coll_series.append(s)
            if include_compare:
                c = _read_year_metrics_compare(year_dir, run_id, group)
                if c is not None and not c.empty:
                    for k, v in cfg_cols.items():
                        c[k] = v
                    coll_compare.append(c)
            if include_permutation:
                pi = _read_year_permutation(year_dir, run_id, group)
                if pi is not None and not pi.empty:
                    for k, v in cfg_cols.items():
                        pi[k] = v
                    coll_perm.append(pi)
            if zscore_mode == "summary":
                zs = _read_year_zscore_summary(year_dir, run_id, group)
                if zs is not None and not zs.empty:
                    for k, v in cfg_cols.items():
                        zs[k] = v
                    coll_zsum.append(zs)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _concat_and_save(frames: List[pd.DataFrame], name: str):
        if not frames:
            if verbose:
                print(f"No data for {group}/{name}, skipping")
            return
        df = pd.concat(frames, ignore_index=True)
        target = out_dir / f"{group}_{name}.csv"
        df.to_csv(target, index=False)
        if verbose:
            print(f"Wrote {target} ({len(df)} rows)")

    if include_metrics:
        _concat_and_save(coll_metrics, "aggregated_metrics")
    if include_compare:
        _concat_and_save(coll_compare, "aggregated_metrics_compare")
    if include_series:
        _concat_and_save(coll_series, "aggregated_series")
    if include_permutation:
        _concat_and_save(coll_perm, "aggregated_permutation_importance")
    if zscore_mode == "summary":
        _concat_and_save(coll_zsum, "aggregated_zscore_summary")


def main():
    parser = argparse.ArgumentParser(description="Aggregate VaR run outputs into compact CSVs per group (base/big)")
    parser.add_argument("--root", type=str, default="outputs", help="Root outputs directory containing base/ and big/")
    parser.add_argument("--out", type=str, default="outputs/aggregated", help="Target output directory for aggregated CSVs")
    parser.add_argument("--groups", type=str, default="base,big", help="Comma-separated groups to process (base,big)")
    parser.add_argument("--no-series", action="store_true", help="Skip aggregating time series (largest file)")
    parser.add_argument("--no-metrics", action="store_true", help="Skip aggregating wf_metrics.csv")
    parser.add_argument("--no-compare", action="store_true", help="Skip aggregating per-period metrics compare")
    parser.add_argument("--no-permutation", action="store_true", help="Skip aggregating permutation importance")
    parser.add_argument("--zscore", type=str, default="summary", choices=["none", "summary"], help="Z-Score aggregation mode")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--make-master", action="store_true", help="Also write master CSVs combining all groups (base+big)")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    verbose = not args.quiet

    for group in groups:
        out_dir = out_root / group
        aggregate_group(
            root=root,
            group=group,
            out_dir=out_dir,
            include_series=not args.no_series,
            include_metrics=not args.no_metrics,
            include_compare=not args.no_compare,
            include_permutation=not args.no_permutation,
            zscore_mode=("none" if args.zscore == "none" else "summary"),
            verbose=verbose,
        )

    # Optionally create master CSVs across all processed groups
    if args.make_master:
        def _try_concat(files: list[Path]) -> Optional[pd.DataFrame]:
            frames: list[pd.DataFrame] = []
            for f in files:
                df = _safe_read_csv(f)
                if df is not None and not df.empty:
                    frames.append(df)
            if not frames:
                return None
            return pd.concat(frames, ignore_index=True)

        master_map = []
        if not args.no_metrics:
            master_map.append(("aggregated_metrics",))
        if not args.no_compare:
            master_map.append(("aggregated_metrics_compare",))
        if not args.no_permutation:
            master_map.append(("aggregated_permutation_importance",))
        if not args.no_series:
            master_map.append(("aggregated_series",))
        if args.zscore != "none":
            master_map.append(("aggregated_zscore_summary",))

        out_root.mkdir(parents=True, exist_ok=True)
        for (name,) in master_map:
            files = [out_root / g / f"{g}_{name}.csv" for g in groups]
            dfm = _try_concat(files)
            if dfm is None or dfm.empty:
                if verbose:
                    print(f"No data found for master {name}, skipping")
                continue
            target = out_root / f"master_{name}.csv"
            dfm.to_csv(target, index=False)
            if verbose:
                print(f"Wrote {target} ({len(dfm)} rows)")


if __name__ == "__main__":
    main()


