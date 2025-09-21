import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def _metric_fullname(metric: str) -> str:
    mapping = {
        'breach_rate': 'Breach‑Rate (Verletzungsrate)',
        'coverage_error': 'Coverage Error',
        'cvar': 'Conditional Value at Risk (CVaR)',
        'rmse': 'Root Mean Squared Error (RMSE)',
        'mae': 'Mean Absolute Error (MAE)',
        'pinball': 'Pinball Loss',
        'brier': 'Brier Score'
    }
    return mapping.get(str(metric), str(metric))


def _read_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / 'config.json'
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _read_wf_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    m_path = run_dir / 'wf_metrics.csv'
    if not m_path.exists():
        return None
    try:
        df = pd.read_csv(m_path)
        # Normalize column names
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception:
        return None


def _extract_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Pick a concise set of parameters to summarize across runs
    keys = [
        'seq_length', 'z_window', 'use_returns', 'use_spx_close', 'refit',
        'timeframe', 'calibration', 'calib_window', 'use_cls_token',
        'layers', 'head_size', 'num_heads', 'ff_dim', 'dropout', 'attn_dropout'
    ]
    out = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg[k]
    # Backward compatibility for names
    if 'z-window' in cfg and 'z_window' not in out:
        out['z_window'] = cfg.get('z-window')
    if 'seq-length' in cfg and 'seq_length' not in out:
        out['seq_length'] = cfg.get('seq-length')
    if 'refit' not in out and 'refit_freq' in cfg:
        out['refit'] = cfg['refit_freq']
    return out


def build_runs_summary(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in sorted(root.glob('run_*')):
        if not run.is_dir():
            continue
        cfg = _read_run_config(run)
        wf = _read_wf_metrics(run)
        if wf is None or wf.empty:
            continue
        # Robust cast
        for col in ['breach_rate','coverage_error','cvar','rmse','mae','pinball','brier','kupiec_p','christoffersen_cc_p']:
            if col in wf.columns:
                wf[col] = pd.to_numeric(wf[col], errors='coerce')

        params = _extract_params(cfg)
        # Aggregate metrics across periods (Mittel + Std)
        def _mean(col: str) -> float:
            return float(wf[col].mean()) if col in wf.columns else float('nan')
        def _std(col: str) -> float:
            return float(wf[col].std(ddof=0)) if col in wf.columns else float('nan')
        agg = {
            'n_periods': int(len(wf)),
            'breach_rate': _mean('breach_rate'),
            'breach_rate_std': _std('breach_rate'),
            'coverage_error': _mean('coverage_error'),
            'coverage_error_std': _std('coverage_error'),
            'cvar': _mean('cvar'),
            'cvar_std': _std('cvar'),
            'rmse': _mean('rmse'),
            'rmse_std': _std('rmse'),
            'mae': _mean('mae'),
            'mae_std': _std('mae'),
            'pinball': _mean('pinball'),
            'pinball_std': _std('pinball'),
            'brier': _mean('brier'),
            'brier_std': _std('brier'),
        }
        # Backtest-Passraten (p>=0.05)
        if 'kupiec_p' in wf.columns:
            n_valid = wf['kupiec_p'].notna().sum()
            agg['kupiec_pass_rate'] = float((wf['kupiec_p'] >= 0.05).sum() / n_valid) if n_valid > 0 else float('nan')
        else:
            agg['kupiec_pass_rate'] = float('nan')
        if 'christoffersen_cc_p' in wf.columns:
            n_valid = wf['christoffersen_cc_p'].notna().sum()
            agg['christoffersen_cc_pass_rate'] = float((wf['christoffersen_cc_p'] >= 0.05).sum() / n_valid) if n_valid > 0 else float('nan')
        else:
            agg['christoffersen_cc_pass_rate'] = float('nan')
        rows.append({'run_dir': str(run), **params, **agg})
    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort by absolute coverage error ascending
    if 'coverage_error' in df.columns:
        df = df.sort_values(by=df['coverage_error'].abs().name, key=lambda s: s.abs())
    return df


def concat_all_wf_metrics(root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run in sorted(root.glob('run_*')):
        m = _read_wf_metrics(run)
        if m is None or m.empty:
            continue
        cfg = _read_run_config(run)
        params = _extract_params(cfg)
        # Robust cast
        for col in ['breach_rate','coverage_error','cvar','rmse','mae','pinball','brier','kupiec_p','christoffersen_cc_p']:
            if col in m.columns:
                m[col] = pd.to_numeric(m[col], errors='coerce')
        m['run_dir'] = str(run)
        # Keine Regime-Flags mehr
        # Parameter anhängen
        for k, v in params.items():
            m[k] = v
        rows.append(m)
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def concat_all_wf_metrics_compare(root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run in sorted(root.glob('run_*')):
        cfg = _read_run_config(run)
        params = _extract_params(cfg)
        for f in run.glob('**/wf_*_metrics_compare.csv'):
            try:
                m = pd.read_csv(f)
            except Exception:
                continue
            if m.empty or 'model' not in m.columns:
                continue
            # Numeric casts
            for col in ['breach_rate','coverage_error','cvar','rmse','mae','pinball','brier','kupiec_p','christoffersen_cc_p']:
                if col in m.columns:
                    m[col] = pd.to_numeric(m[col], errors='coerce')
            # Year
            if 'period' in m.columns:
                try:
                    m['year'] = m['period'].astype(str).str.extract(r'(\d{4})', expand=False).astype(int)
                except Exception:
                    m['year'] = np.nan
            m['run_dir'] = str(run)
            for k, v in params.items():
                m[k] = v
            rows.append(m)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _add_model_head_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # d_model-Heuristik
    if 'head_size' in out.columns and 'num_heads' in out.columns:
        out['d_model'] = pd.to_numeric(out['head_size'], errors='coerce') * pd.to_numeric(out['num_heads'], errors='coerce')
    else:
        out['d_model'] = np.nan
    # Modellgröße anhand Kapazität
    def _model_size(row: pd.Series) -> str:
        try:
            d = float(row.get('d_model', np.nan))
            layers = float(row.get('layers', np.nan))
            if not np.isnan(d) and not np.isnan(layers) and d >= 512 and layers >= 6:
                return 'Big'
        except Exception:
            pass
        return 'Base'
    out['model_size'] = out.apply(_model_size, axis=1)
    # Head-Variante
    def _head(row: pd.Series) -> str:
        v = row.get('use_cls_token', True)
        try:
            return 'CLS' if bool(v) else 'Last'
        except Exception:
            return 'CLS'
    out['head'] = out.apply(_head, axis=1)
    return out


def plot_pvalue_hist_ecdf(all_wf: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for col in ['kupiec_p', 'christoffersen_cc_p']:
        if col not in all_wf.columns:
            continue
        data = pd.to_numeric(all_wf[col], errors='coerce').dropna()
        if data.empty:
            continue
        # Histogramm
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=20, color='tab:gray', edgecolor='k')
        ax.axvline(0.05, color='red', linestyle='--', label='0.05')
        title = 'Kupiec-Test p-Werte' if col == 'kupiec_p' else 'Christoffersen CC-Test p-Werte'
        ax.set_title(title)
        ax.set_xlabel('p-Wert'); ax.set_ylabel('Häufigkeit'); ax.legend()
        plt.tight_layout(); plt.savefig(outdir / f'pvalue_hist_{col}.png', dpi=150); plt.close(fig)
        # ECDF
        x = np.sort(data.values)
        y = np.arange(1, len(x)+1) / len(x)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color='tab:blue')
        ax.axvline(0.05, color='red', linestyle='--')
        ax.set_title((title + ' – ECDF'))
        ax.set_xlabel('p-Wert'); ax.set_ylabel('F(x)')
        plt.tight_layout(); plt.savefig(outdir / f'pvalue_ecdf_{col}.png', dpi=150); plt.close(fig)


def plot_tradeoff_scatter(summary_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = _add_model_head_cols(summary_df)
    if 'coverage_error' not in df.columns or 'pinball' not in df.columns:
        return
    x = df['coverage_error'].abs()
    y = df['pinball']
    colors = df['model_size'].map({'Base':'tab:green','Big':'tab:purple'}).fillna('tab:gray')
    markers = df['head'].map({'CLS':'o','Last':'s'}).fillna('o')
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in markers.unique():
        sel = markers == mk
        ax.scatter(x[sel], y[sel], c=colors[sel], marker=str(mk), edgecolor='k', alpha=0.8, label=f'Head {"CLS" if mk=="o" else "Last"}')
    ax.set_xlabel('Absoluter Coverage Error'); ax.set_ylabel('Pinball Loss')
    ax.set_title('Trade-off: Absoluter Coverage Error vs. Pinball Loss')
    plt.tight_layout(); plt.savefig(outdir / 'tradeoff_ce_vs_pinball.png', dpi=150); plt.close(fig)


def correlations_table(summary_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = _add_model_head_cols(summary_df)
    # numerische Prädiktoren
    num_cols = []
    for c in ['seq_length','z_window','layers','d_model','ff_dim','dropout','attn_dropout']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            num_cols.append(c)
    metric_cols = [c for c in ['coverage_error','breach_rate','pinball','cvar','rmse','mae','brier'] if c in df.columns]
    corrs = []
    for m in metric_cols:
        for p in num_cols:
            s = df[[m, p]].dropna()
            if s.empty:
                continue
            # Skip pairs with no variance to avoid ConstantInputWarning
            if s[m].nunique(dropna=True) <= 1 or s[p].nunique(dropna=True) <= 1:
                continue
            rho, pv = stats.spearmanr(s[m], s[p])
            corrs.append({'metric': m, 'param': p, 'spearman_rho': rho, 'p_value': pv, 'n': len(s)})
    if corrs:
        pd.DataFrame(corrs).to_csv(outdir / 'correlations_spearman.csv', index=False)


def ols_anova_reports(summary_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = _add_model_head_cols(summary_df)
    for col in ['seq_length','z_window']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['coverage_error','seq_length','z_window'])
    if df.empty:
        return
    # OLS für coverage_error
    try:
        model = smf.ols('coverage_error ~ C(model_size) + C(head) + seq_length + z_window', data=df).fit()
        txt = model.summary().as_text()
        with open(outdir / 'ols_coverage_error.txt', 'w') as f:
            f.write(txt)
        # Zusätzlich PNG erzeugen
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.text(0.01, 0.99, 'OLS: Coverage Error ~ Modellgröße + Head + Sequenzlänge + Z‑Fenster', transform=ax.transAxes, fontsize=10, va='top')
        ax.text(0.01, 0.95, txt, family='monospace', transform=ax.transAxes, fontsize=8, va='top')
        plt.tight_layout()
        plt.savefig(outdir / 'ols_coverage_error.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        with open(outdir / 'ols_coverage_error.txt', 'w') as f:
            f.write(f'OLS fehlgeschlagen: {e}')


def aggregate_feature_importance(root: Path, outdir: Path, top_k: int = 20) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[pd.DataFrame] = []
    for run in sorted(root.glob('run_*')):
        # Sammle alle per-Period PI CSVs
        for f in run.glob('**/wf_*_permutation_importance.csv'):
            try:
                df = pd.read_csv(f)
                if 'feature' in df.columns and 'pinball_degradation' in df.columns:
                    df['run_dir'] = str(run)
                    rows.append(df[['feature','pinball_degradation','run_dir']])
            except Exception:
                pass
    if not rows:
        return
    pi = pd.concat(rows, ignore_index=True)
    pi['feature'] = pi['feature'].astype(str)
    pi['pinball_degradation'] = pd.to_numeric(pi['pinball_degradation'], errors='coerce')
    pi = pi.dropna(subset=['pinball_degradation'])
    agg = pi.groupby('feature', as_index=False)['pinball_degradation'].mean().sort_values('pinball_degradation', ascending=False)
    agg.to_csv(outdir / 'feature_importance_overall.csv', index=False)
    # Top-K Barplot
    top = agg.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top['feature'], top['pinball_degradation'], color='tab:gray')
    ax.set_xlabel('Durchschn. Pinball-Degradation'); ax.set_ylabel('Feature')
    ax.set_title('Permutation Importance – Top Features (über alle Runs)')
    plt.tight_layout(); plt.savefig(outdir / 'feature_importance_topk.png', dpi=150); plt.close(fig)


def plot_metric_bars_by_year(all_wf: pd.DataFrame, outdir: Path, metrics: List[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if 'period' not in all_wf.columns:
        return
    import re
    df = all_wf.copy()
    df['year'] = df['period'].astype(str).apply(lambda s: int(re.search(r'(\d{4})', s).group(1)) if re.search(r'(\d{4})', s) else np.nan)
    df = df.dropna(subset=['year'])
    if df.empty:
        return
    # Nur tatsächlich vorhandene OOS-Jahre als integer-sorted Achse
    df['year'] = df['year'].astype(int)
    year_group = df.groupby('year', sort=True)
    for m in metrics:
        if m not in df.columns:
            continue
        g = year_group[m].mean().sort_index()
        if g.empty:
            continue
        years = g.index.astype(int).tolist()
        x = np.arange(len(years))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x, g.values, color='tab:blue')
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years])
        ax.set_xlabel('Jahr'); ax.set_ylabel(_metric_fullname(m))
        ax.set_title(f'{_metric_fullname(m)} nach Jahr (Mittel über Runs)')
        plt.tight_layout(); plt.savefig(outdir / f'metrics_by_year_{m}.png', dpi=150); plt.close(fig)


def _run_group_analysis(summary_df: pd.DataFrame, all_wf: Optional[pd.DataFrame], outdir: Path, all_wf_compare: Optional[pd.DataFrame] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    # Persist group summary
    summary_df.to_csv(outdir / 'runs_summary.csv', index=False)
    metric_cols = [c for c in ['coverage_error', 'breach_rate', 'cvar', 'pinball', 'rmse', 'mae'] if c in summary_df.columns]
    if metric_cols:
        plot_heatmaps(summary_df, outdir, metric_cols)
    plot_scatter_ce(summary_df, outdir)
    plot_breach_rate_bars(summary_df, outdir)
    correlations_table(summary_df, outdir)
    ols_anova_reports(summary_df, outdir)
    if all_wf is not None and not all_wf.empty:
        plot_pvalue_hist_ecdf(all_wf, outdir)
        plot_metric_bars_by_year(all_wf, outdir, metrics=metric_cols)
    if all_wf_compare is not None and not all_wf_compare.empty:
        compare_metrics = [c for c in ['coverage_error', 'breach_rate', 'cvar', 'pinball', 'rmse', 'mae'] if c in all_wf_compare.columns]
        compare_summary_tables(all_wf_compare, outdir, compare_metrics)
        plot_compare_overall_bars(all_wf_compare, outdir, compare_metrics)
        plot_compare_by_year_bars(all_wf_compare, outdir, compare_metrics)


def split_group_analyze(summary_df: pd.DataFrame, all_wf: Optional[pd.DataFrame], base_outdir: Path, all_wf_compare: Optional[pd.DataFrame] = None) -> None:
    # Ergänze abgeleitete Spalten
    df_ext = _add_model_head_cols(summary_df)
    all_wf_ext = _add_model_head_cols(all_wf) if all_wf is not None and not all_wf.empty else None
    compare_ext = _add_model_head_cols(all_wf_compare) if all_wf_compare is not None and not all_wf_compare.empty else None
    model_sizes = sorted([ms for ms in df_ext['model_size'].dropna().unique()]) if 'model_size' in df_ext.columns else []
    heads = sorted([h for h in df_ext['head'].dropna().unique()]) if 'head' in df_ext.columns else []
    # Falls keine expliziten Größen erkannt wurden, alles als "Base" behandeln
    if not model_sizes:
        model_sizes = ['Base']
        df_ext['model_size'] = 'Base'
        if all_wf_ext is not None:
            all_wf_ext['model_size'] = 'Base'
    if not heads:
        heads = ['CLS']
        df_ext['head'] = 'CLS'
        if all_wf_ext is not None:
            all_wf_ext['head'] = 'CLS'
    for ms in model_sizes:
        ms_dir = base_outdir / ms.lower()
        for hd in heads:
            hd_label = 'cls-head' if hd == 'CLS' else 'last-token'
            grp_dir = ms_dir / hd_label
            sub_sum = df_ext[(df_ext['model_size'] == ms) & (df_ext['head'] == hd)].copy()
            if sub_sum.empty:
                continue
            if all_wf_ext is not None:
                sub_wf = all_wf_ext[(all_wf_ext['model_size'] == ms) & (all_wf_ext['head'] == hd)].copy()
            else:
                sub_wf = None
            # Group-specific compare df (Transformer vs. Baselines) für diese Head/Modellgröße
            if compare_ext is not None and 'model_size' in compare_ext.columns and 'head' in compare_ext.columns:
                sub_cmp = compare_ext[(compare_ext['model_size'] == ms) & (compare_ext['head'] == hd)].copy()
            else:
                sub_cmp = None
            _run_group_analysis(sub_sum, sub_wf, grp_dir, sub_cmp)


def compare_summary_tables(all_wf_compare: pd.DataFrame, outdir: Path, metrics: List[str]) -> None:
    # Aggregierte Kennzahlen je Modell
    agg_rows = []
    for model, grp in all_wf_compare.groupby('model'):
        row = {'model': model, 'n_periods': int(len(grp))}
        for m in metrics:
            row[f'{m}_mean'] = float(pd.to_numeric(grp[m], errors='coerce').mean())
            row[f'{m}_std'] = float(pd.to_numeric(grp[m], errors='coerce').std(ddof=0))
        # Passraten
        if 'kupiec_p' in all_wf_compare.columns:
            valid = grp['kupiec_p'].notna()
            row['kupiec_pass_rate'] = float((grp.loc[valid, 'kupiec_p'] >= 0.05).mean()) if valid.any() else float('nan')
        if 'christoffersen_cc_p' in all_wf_compare.columns:
            valid = grp['christoffersen_cc_p'].notna()
            row['christoffersen_cc_pass_rate'] = float((grp.loc[valid, 'christoffersen_cc_p'] >= 0.05).mean()) if valid.any() else float('nan')
        agg_rows.append(row)
    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(outdir / 'compare_summary.csv', index=False)


def _model_color(model: str) -> str:
    cmap = {
        'Transformer': 'tab:blue',
        'GARCH': 'tab:orange',
        'LSTM': 'tab:green'
    }
    return cmap.get(model, 'tab:gray')


def plot_compare_overall_bars(all_wf_compare: pd.DataFrame, outdir: Path, metrics: List[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    models = [m for m in ['Transformer','GARCH','LSTM'] if m in set(all_wf_compare['model'])]
    if not models:
        return
    for met in metrics:
        g = all_wf_compare.groupby('model')[met].mean().reindex(models)
        if g.dropna().empty:
            continue
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [_model_color(m) for m in models]
        ax.bar(x, g.values, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylabel(_metric_fullname(met))
        ax.set_title(f'{_metric_fullname(met)} – Modellvergleich (Mittel über Jahre)')
        plt.tight_layout(); plt.savefig(outdir / f'compare_overall_{met}.png', dpi=150); plt.close(fig)


def plot_compare_by_year_bars(all_wf_compare: pd.DataFrame, outdir: Path, metrics: List[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if 'year' not in all_wf_compare.columns:
        return
    models = [m for m in ['Transformer','GARCH','LSTM'] if m in set(all_wf_compare['model'])]
    if not models:
        return
    for met in metrics:
        pivot = all_wf_compare.pivot_table(index='year', columns='model', values=met, aggfunc='mean').reindex(columns=models)
        if pivot.dropna(how='all').empty:
            continue
        years = pivot.index.astype(int).tolist()
        x = np.arange(len(years))
        width = 0.18
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, model in enumerate(models):
            vals = pivot[model].values
            ax.bar(x + (i - (len(models)-1)/2) * width, vals, width=width, label=model, color=_model_color(model))
        ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
        ax.set_xlabel('Jahr'); ax.set_ylabel(_metric_fullname(met))
        ax.set_title(f'{_metric_fullname(met)} – Vergleich nach Jahr')
        ax.legend()
        plt.tight_layout(); plt.savefig(outdir / f'compare_by_year_{met}.png', dpi=150); plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, outdir: Path, metric_cols: List[str], index_col: str = 'seq_length', col_col: str = 'z_window') -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Keep only necessary columns and drop incomplete rows
    sub = df[[index_col, col_col] + metric_cols].dropna(subset=[index_col, col_col])
    # Cast to numeric where possible
    sub[index_col] = pd.to_numeric(sub[index_col], errors='coerce')
    sub[col_col] = pd.to_numeric(sub[col_col], errors='coerce')
    sub = sub.dropna(subset=[index_col, col_col])
    for m in metric_cols:
        pivot = sub.pivot_table(index=index_col, columns=col_col, values=m, aggfunc='mean')
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, aspect='auto', cmap='coolwarm')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_ylabel('Sequenzlänge')
        ax.set_xlabel('Z‑Score‑Fenster')
        ax.set_title(f'{_metric_fullname(m)} über Sequenzlänge × Z‑Score‑Fenster')
        fig.colorbar(im, ax=ax, label=_metric_fullname(m))
        plt.tight_layout()
        plt.savefig(outdir / f'heatmap_{m}.png', dpi=150)
        plt.close(fig)


def plot_scatter_ce(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sub = df.copy()
    sub['z_window'] = pd.to_numeric(sub.get('z_window', np.nan), errors='coerce')
    sub['coverage_error'] = pd.to_numeric(sub.get('coverage_error', np.nan), errors='coerce')
    sub['cvar'] = pd.to_numeric(sub.get('cvar', np.nan), errors='coerce')
    sub['seq_length'] = pd.to_numeric(sub.get('seq_length', np.nan), errors='coerce')
    if sub[['z_window', 'coverage_error']].dropna().empty:
        return
    size = 200 * (sub['cvar'].abs() / (sub['cvar'].abs().max() if sub['cvar'].abs().max() > 0 else 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(sub['z_window'], sub['coverage_error'], s=size, color='tab:blue', alpha=0.8, edgecolor='k')
    ax.axhline(0.0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Z‑Score‑Fenster')
    ax.set_ylabel('Coverage Error')
    ax.set_title('Z‑Score‑Fenster vs. Coverage Error (Markergröße ~ |CVaR|)')
    # Sequenzlänge beschriften
    try:
        for _, r in sub.dropna(subset=['z_window','coverage_error','seq_length']).iterrows():
            ax.annotate(f"L{int(r['seq_length'])}", (r['z_window'], r['coverage_error']), textcoords='offset points', xytext=(4,4), fontsize=8)
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(outdir / 'scatter_zwindow_vs_coverage_error.png', dpi=150)
    plt.close(fig)


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    denominator = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    radius = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    low = (center - radius) / denominator
    high = (center + radius) / denominator
    return (low, high)


def plot_breach_rate_bars(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sub = df.copy()
    sub['breach_rate'] = pd.to_numeric(sub.get('breach_rate', np.nan), errors='coerce')
    if sub['breach_rate'].dropna().empty:
        return
    # Beschriftung ohne Refit/Calibration, wissenschaftlich ausgeschrieben
    def _label(row: pd.Series) -> str:
        seq = row.get('seq_length', '')
        zw = row.get('z_window', '')
        return f"Sequenzlänge {seq} – Z‑Score‑Fenster {zw}"
    labels = [_label(r) for _, r in sub.iterrows()]
    # Approximate N via typical daily year length across WF periods; fallback n=252
    n = 252
    cis = [wilson_ci(br, n) for br in sub['breach_rate'].fillna(0.0)]
    lows = [br - ci[0] for br, ci in zip(sub['breach_rate'], cis)]
    highs = [ci[1] - br for br, ci in zip(sub['breach_rate'], cis)]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, sub['breach_rate'], yerr=[lows, highs], color='tab:blue', alpha=0.7, capsize=3)
    ax.axhline(0.05, color='red', linestyle='--', label='Ziel q = 0,05')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Breach‑Rate (Verletzungsrate) – Wilson‑CI 95%')
    ax.set_title('Breach‑Rate mit 95%-Wilson‑Konfidenzintervall pro Run')
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'breach_rate_ci_bars.png', dpi=150)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='outputs', help='Wurzelverzeichnis mit run_* Ordnern')
    parser.add_argument('--out', type=str, default='outputs/runs_dashboard', help='Zielordner für Zusammenfassung und Plots')
    # entfernt: --highlight-year
    parser.add_argument('--runs-subdir', type=str, default=None, help='Optional: Unterordner wie outputs/base zur Filterung (setzt root entsprechend)')
    args = parser.parse_args()

    root = Path(args.runs_subdir) if args.runs_subdir else Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_runs_summary(root)
    if df.empty:
        print('Keine lauffähigen Runs gefunden (config.json + wf_metrics.csv).')
        return
    df.to_csv(outdir / 'runs_summary.csv', index=False)

    # Keine Top‑Level‑Aggregationsplots mehr (um Heads nicht zu vermischen).
    # Hierarchische Auswertung: Base/Big -> Head -> Seq/Z
    all_wf = concat_all_wf_metrics(root)
    all_wf_compare = concat_all_wf_metrics_compare(root)
    # Gruppenspezifische Analysen inklusive Benchmark‑Vergleichen (pro Head/Größe)
    split_group_analyze(df, all_wf, outdir, all_wf_compare)
    print(f'Runs‑Dashboard erstellt unter: {outdir}')


if __name__ == '__main__':
    main()


