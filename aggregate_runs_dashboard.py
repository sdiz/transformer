import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def _context_from_path(base: Path, file_path: Path) -> dict:
    rel = file_path.relative_to(base)
    parts = list(rel.parts)
    # Expected patterns:
    # base/{group}/runs_summary.csv
    # base/{group}/{model_size}/{head}/runs_summary.csv
    ctx = {
        'group_dir': None,
        'model_size_dir': None,
        'head_dir': None,
        'dashboard_scope': 'global',
        'source_path': str(file_path),
    }
    if len(parts) >= 2:
        ctx['group_dir'] = parts[0]
        # If deeper structure exists
        if len(parts) >= 4:
            ctx['model_size_dir'] = parts[1]
            ctx['head_dir'] = parts[2]
            ctx['dashboard_scope'] = 'head'
        else:
            ctx['dashboard_scope'] = 'group'
    return ctx


def _concat_csvs(base: Path, pattern: str, out_path: Path) -> None:
    rows: List[pd.DataFrame] = []
    for f in base.rglob(pattern):
        df = _safe_read_csv(f)
        if df is None or df.empty:
            continue
        ctx = _context_from_path(base, f)
        for k, v in ctx.items():
            df[k] = v
        rows.append(df)
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Aggregate runs_dashboard CSVs into master CSVs')
    parser.add_argument('--root', type=str, default='outputs/runs_dashboard', help='Dashboard root path')
    parser.add_argument('--out', type=str, default='outputs/runs_dashboard', help='Where to write master CSVs')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    args = parser.parse_args()

    base = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Master over all heads/groups
    _concat_csvs(base, 'runs_summary.csv', out / 'master_runs_summary.csv')
    _concat_csvs(base, 'compare_summary.csv', out / 'master_compare_summary.csv')
    _concat_csvs(base, 'correlations_spearman.csv', out / 'master_correlations_spearman.csv')

    if not args.quiet:
        print('Master dashboard CSVs written to', out)


if __name__ == '__main__':
    main()


