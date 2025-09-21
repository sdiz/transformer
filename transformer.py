import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import math
from typing import Optional, Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
import warnings
import os
import random
import argparse
from pathlib import Path
import json
import hashlib
import logging
import sys

try:
    from scipy.stats import chi2, t as student_t
except Exception:
    chi2 = None
    student_t = None

# FRED API (fredapi) – erforderlich
from fredapi import Fred  # type: ignore

# ----------------------------- Quantile Loss -----------------------------
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
    return loss

# ----------------------------- Datenabruf --------------------------------
def prepare_macro_and_market_data(
    start='1980-01-01',
    end=None,
    release_aware: bool = False,
    macro_release_lag_days: int = 10,
    cache_dir: str = 'data',
    use_cache: bool = True,
    refresh_cache: bool = False,
    fred_api_key: Optional[str] = None,
):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # Cache-Pfade
    cache_path = Path(cache_dir)
    daily_cache = cache_path / 'combined_daily.csv'
    weekly_cache = cache_path / 'combined_weekly.csv'
    monthly_cache = cache_path / 'combined_monthly.csv'

    if use_cache and not refresh_cache and daily_cache.exists() and weekly_cache.exists() and monthly_cache.exists():
        try:
            daily_df = pd.read_csv(daily_cache)
            weekly_df = pd.read_csv(weekly_cache)
            monthly_df = pd.read_csv(monthly_cache)
            for df in (daily_df, weekly_df, monthly_df):
                if 'Unnamed: 0' in df.columns:
                    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
            # Zeitraumbegrenzung auch bei Cache-Load anwenden
            start_dt = pd.to_datetime(start) if start is not None else None
            end_dt = pd.to_datetime(end) if end is not None else None
            if start_dt is not None:
                daily_df = daily_df.loc[start_dt:]
                weekly_df = weekly_df.loc[start_dt:]
                monthly_df = monthly_df.loc[start_dt:]
            if end_dt is not None:
                daily_df = daily_df.loc[:end_dt]
                weekly_df = weekly_df.loc[:end_dt]
                monthly_df = monthly_df.loc[:end_dt]
            # Keine globale Eliminierung von Zeilen via dropna beim Cache-Load
            return daily_df, weekly_df, monthly_df
        except Exception:
            pass  # Fallback auf Neuaufbau

    fred_series = {
        # Preise/Inflation
        'cpi_mom': 'CPIAUCSL',
        'pce_price_index': 'PCEPI',
        # Arbeitsmarkt/Konjunktur
        'unemployment_rate': 'UNRATE',
        'nonfarm_payrolls': 'PAYEMS',
        'industrial_production': 'INDPRO',
        'capacity_utilization': 'TCU',
        # Housing
        'housing_starts': 'HOUST',
        'building_permits': 'PERMIT',
        # Konsum/Bestellungen/Vertrieb
        'retail_sales': 'RSAFS',
        'durable_orders': 'DGORDER',
        'consumer_confidence': 'UMCSENT',
        # Geldpolitik / Policy
        'fed_funds': 'FEDFUNDS',
        # Risikoindikatoren (aus risk_indicators.py)
        't10y2y_spread': 'T10Y2Y',
        't10y3m_spread': 'T10Y3M',
        'hy_index': 'BAMLH0A0HYM2EY',
        'ig_index': 'BAMLC0A4CBBBEY',
        'hy_ccc_spread': 'BAMLH0A3HYC',
        # Unsicherheit
        'epu_index': 'USEPUINDXD',
    }

    macro_monthly = pd.DataFrame()
    loaded_series = []
    missing_series = []
    # FRED-Client initialisieren (API-Key erforderlich)
    if not fred_api_key:
        raise RuntimeError("FRED_API_KEY nicht gesetzt. Bitte Umgebungsvariable FRED_API_KEY setzen.")
    try:
        fred_client = Fred(api_key=fred_api_key)
    except Exception as e:
        raise RuntimeError(f"FRED API Initialisierung fehlgeschlagen: {e}")

    for name, code in fred_series.items():
        ts_m = None
        try:
            ts = fred_client.get_series(code)
            if ts is not None and len(ts) > 0:
                ts = ts.dropna()
                # Auf Zeitraum begrenzen
                if start is not None:
                    ts = ts.loc[pd.to_datetime(start):]
                if end is not None:
                    ts = ts.loc[:pd.to_datetime(end)]
                # Monatsende vereinheitlichen
                if not isinstance(ts.index, pd.DatetimeIndex):
                    ts.index = pd.to_datetime(ts.index)
                ts_m = ts.resample('ME').last()
        except Exception:
            ts_m = None
        if ts_m is not None and len(ts_m) > 0:
            macro_monthly[name] = ts_m
            loaded_series.append(name)
        else:
            missing_series.append(name)

    macro_monthly.index = macro_monthly.index.to_period('M').to_timestamp('M')
    # CPI: convert level (CPIAUCSL) to MoM inflation rate (1-month percent change)
    if 'cpi_mom' in macro_monthly.columns:
        try:
            cpi_series = macro_monthly['cpi_mom'].astype(float)
            macro_monthly['cpi_mom'] = cpi_series.pct_change(1)
        except Exception:
            pass
    # Release-aware Approximation: Werte ab (Monatsende + Lag) gültig
    if release_aware:
        macro_shifted = macro_monthly.copy()
        macro_shifted.index = macro_shifted.index + pd.Timedelta(days=macro_release_lag_days)
        macro_daily = macro_shifted.resample('D').ffill()
        macro_weekly = macro_shifted.resample('W-FRI').ffill()
    else:
        macro_daily = macro_monthly.resample('D').ffill()
        macro_weekly = macro_monthly.resample('W-FRI').ffill()

    spx = yf.download('^GSPC', start=start, end=end, auto_adjust=True)

    market_daily = pd.DataFrame(index=spx.index)
    market_daily['spx_close'] = spx['Close']
    market_daily['returns'] = spx['Close'].pct_change()
    market_daily = market_daily.dropna()

    # Corporate Bond Spread (High Yield - Investment Grade) berechnen
    if {'hy_index', 'ig_index'}.issubset(macro_daily.columns):
        macro_daily['corp_bond_spread'] = macro_daily['hy_index'] - macro_daily['ig_index']
        macro_weekly['corp_bond_spread'] = macro_weekly['hy_index'] - macro_weekly['ig_index']
        macro_monthly['corp_bond_spread'] = macro_monthly['hy_index'] - macro_monthly['ig_index']

    combined_daily = market_daily.join(macro_daily, how='left')

    market_weekly = market_daily.resample('W-FRI').agg({
        'spx_close': 'last',
        'returns': 'sum',
    })
    combined_weekly = market_weekly.join(macro_weekly, how='left')

    market_monthly = market_daily.resample('ME').agg({
        'spx_close': 'last',
        'returns': 'sum',
    })
    combined_monthly = market_monthly.join(macro_monthly, how='left')

    # Sicherheits-Trim auf Start/Ende
    start_dt = pd.to_datetime(start) if start is not None else None
    end_dt = pd.to_datetime(end) if end is not None else None
    if start_dt is not None:
        combined_daily = combined_daily.loc[start_dt:]
        combined_weekly = combined_weekly.loc[start_dt:]
        combined_monthly = combined_monthly.loc[start_dt:]
    if end_dt is not None:
        combined_daily = combined_daily.loc[:end_dt]
        combined_weekly = combined_weekly.loc[:end_dt]
        combined_monthly = combined_monthly.loc[:end_dt]

    # Cache schreiben
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        combined_daily.to_csv(daily_cache, index_label='Date')
        combined_weekly.to_csv(weekly_cache, index_label='Date')
        combined_monthly.to_csv(monthly_cache, index_label='Date')
    except Exception as e:
        print(f"Warnung: Konnte CSV-Cache nicht schreiben: {e}")

    if missing_series:
        warnings.warn(
            f"Folgende Makroserien konnten nicht geladen werden und fehlen im Datensatz: {sorted(missing_series)}."
        )
    # Wichtig: Nicht global dropna – Training filtert nach gewählten Features
    return combined_daily, combined_weekly, combined_monthly

# ---------------------- Chronologische Splits ----------------------------
def split_chronologically(df, train_start, train_end, val_end, test_end=None):
    if test_end is None:
        test_end = df.index.max()
    df = df.sort_index()
    train_df = df.loc[(df.index >= train_start) & (df.index <= train_end)]
    val_df = df.loc[(df.index > train_end) & (df.index <= val_end)]
    test_df = df.loc[(df.index > val_end) & (df.index <= test_end)]
    return train_df, val_df, test_df

# ---------------------- Sequenzen erstellen -----------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

# ---------------------- Gemeinsame Vorverarbeitung ------------------------
def prepare_design_matrix(df: pd.DataFrame, features: list, z_window: int, use_zscore: bool) -> tuple[pd.DataFrame, pd.Series]:
    df_clean = df.copy()
    target = df_clean['returns'].dropna()
    feats = [f for f in features if f in df_clean.columns]
    if len(feats) == 0:
        raise ValueError("Keine Features für Design-Matrix gefunden.")
    X = df_clean[feats].loc[target.index]
    X = X.ffill()
    if use_zscore:
        X = rolling_zscore(X, window=z_window)
    # Aligniere auf returns; entferne nur verbleibende NaNs (Modellverträglichkeit)
    common = X.index.intersection(target.index)
    X = X.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
    target = target.loc[X.index]
    return X, target

def build_sequences_from_df(X_df: pd.DataFrame, y: pd.Series, seq_length: int) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    X_np = X_df.values.astype(np.float32)
    y_np = y.values.astype(np.float32)
    X_seq, _ = create_sequences(X_np, seq_length)
    _, y_seq = create_sequences(y_np.reshape(-1, 1), seq_length)
    X_seq = X_seq.reshape((X_seq.shape[0], seq_length, X_seq.shape[-1]))
    idx = X_df.index[seq_length:]
    return X_seq, y_seq, idx

# ---------------------- Feature-History-Filter -----------------------------
def filter_features_by_history(df: pd.DataFrame, feature_cols: list, min_len: int) -> tuple[list, list]:
    """Wähle nur Features mit ausreichender Historie aus.

    - min_len: erforderliche Anzahl verwertbarer Beobachtungen (nach ffill, vor z-Score)
    - returns bleibt immer erhalten, falls vorhanden
    """
    good: list = []
    bad: list = []
    for f in feature_cols:
        if f not in df.columns:
            continue
        if f == 'returns':
            good.append(f)
            continue
        s = df[f].ffill().dropna()
        if len(s) >= min_len:
            good.append(f)
        else:
            bad.append(f)
    return good, bad

# ---------------------- Loss- und Test-Helfer ----------------------------
def pinball_loss_series(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> np.ndarray:
    e = y_true - y_pred
    return np.maximum(q * e, (q - 1) * e)

def _safe_chi2_sf(x: float, df: int) -> float:
    if chi2 is None:
        return float('nan')
    return float(chi2.sf(x, df))

def kupiec_test(breaches: np.ndarray, alpha: float) -> Tuple[float, float]:
    n = len(breaches)
    x = int(breaches.sum())
    if n == 0:
        return float('nan'), float('nan')
    pi = x / n if n > 0 else 0.0
    # Vermeide log(0)
    pi = min(max(pi, 1e-10), 1 - 1e-10)
    term1 = (1 - alpha) ** (n - x) * (alpha ** x)
    term2 = (1 - pi) ** (n - x) * (pi ** x)
    term1 = max(term1, 1e-300)
    term2 = max(term2, 1e-300)
    LR_uc = -2.0 * math.log(term1 / term2)
    p_value = _safe_chi2_sf(LR_uc, df=1)
    return LR_uc, p_value

def christoffersen_test(breaches: np.ndarray, alpha: float) -> Tuple[float, float, float, float]:
    # Übergangswahrscheinlichkeiten 2x2
    b = breaches.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(b)):
        if b[i-1] == 0 and b[i] == 0:
            n00 += 1
        elif b[i-1] == 0 and b[i] == 1:
            n01 += 1
        elif b[i-1] == 1 and b[i] == 0:
            n10 += 1
        elif b[i-1] == 1 and b[i] == 1:
            n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0
    # Likelihoods
    def _l(p0, p1):
        def _pow(p, k):
            p = min(max(p, 1e-10), 1 - 1e-10)
            return p ** k if k > 0 else 1.0
        return _pow(1 - p0, n00) * _pow(p0, n01) * _pow(1 - p1, n10) * _pow(p1, n11)
    L0 = _l(pi, pi)
    L1 = _l(pi0, pi1)
    L0 = max(L0, 1e-300)
    L1 = max(L1, 1e-300)
    LR_ind = -2.0 * math.log(L0 / L1)
    p_ind = _safe_chi2_sf(LR_ind, df=1)
    # Kombiniert: UC (gegen alpha) + IND
    # UC aus kupiec bereits berechnet werden könnte; hier neu mit pi/alpha
    # Formale CC-Statistik
    # Likelihood unter H0 UC (pi=alpha)
    b_sum = n01 + n11
    n_total = n0 + n1
    term_uc_h0 = (1 - alpha) ** (n_total - b_sum) * (alpha ** b_sum)
    term_uc_h1 = (1 - pi) ** (n_total - b_sum) * (pi ** b_sum)
    term_uc_h0 = max(term_uc_h0, 1e-300)
    term_uc_h1 = max(term_uc_h1, 1e-300)
    LR_uc = -2.0 * math.log(term_uc_h0 / term_uc_h1)
    LR_cc = LR_uc + LR_ind
    p_cc = _safe_chi2_sf(LR_cc, df=2)
    return LR_ind, p_ind, LR_cc, p_cc

# ---------------------- Kalibrierung & Metriken Utils ---------------------
def compute_scale_from_grid(y_cal: np.ndarray, preds_cal: np.ndarray, q_target: float, scale_min: float, scale_max: float, steps: int = 61) -> float:
    grid = np.linspace(scale_min, scale_max, steps)
    coverages = [np.mean(y_cal < s * preds_cal) for s in grid]
    idx_best = int(np.argmin(np.abs(np.array(coverages) - q_target)))
    return float(grid[idx_best])

def compute_conformal_shift(y_cal: np.ndarray, preds_cal: np.ndarray, q_target: float) -> float:
    return float(np.quantile(y_cal - preds_cal, q_target))

def aggregate_metrics_dict(y_true: np.ndarray, preds: np.ndarray, q: float) -> Dict[str, float]:
    breaches = y_true < preds
    breach_rate = float(np.mean(breaches))
    cvar = float(y_true[breaches].mean()) if np.any(breaches) else float('nan')
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    mae = float(np.mean(np.abs(y_true - preds)))
    pinball = float(pinball_loss_series(y_true, preds, q).mean())
    brier = float(np.mean((breaches.astype(float) - q) ** 2))
    LR_uc, p_uc = kupiec_test(breaches, alpha=q)
    LR_ind, p_ind, LR_cc, p_cc = christoffersen_test(breaches, alpha=q)
    return {
        'breach_rate': breach_rate,
        'coverage_error': breach_rate - q,
        'cvar': cvar,
        'rmse': rmse,
        'mae': mae,
        'pinball': pinball,
        'brier': brier,
        'kupiec_LR': LR_uc,
        'kupiec_p': p_uc,
        'christoffersen_ind_LR': LR_ind,
        'christoffersen_ind_p': p_ind,
        'christoffersen_cc_LR': LR_cc,
        'christoffersen_cc_p': p_cc,
    }

def add_compare_metrics(records: list, period: str, model_name: str, y_true: np.ndarray, preds: Optional[np.ndarray], q: float) -> None:
    if preds is None:
        return
    valid = ~np.isnan(preds)
    yb, vb = y_true[valid], preds[valid]
    if len(yb) <= 5:
        return
    m = aggregate_metrics_dict(yb, vb, q)
    m.update({'period': period, 'model': model_name})
    records.append(m)

def save_figure(fig, path: Path) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    except Exception as e:
        print(f"⚠️  Konnte Plot nicht speichern ({e})")

def newey_west_var(d: np.ndarray, max_lag: Optional[int] = None) -> float:
    T = len(d)
    if T <= 1:
        return float('nan')
    if max_lag is None:
        max_lag = int(T ** (1/3))
    mu = d.mean()
    gamma0 = np.mean((d - mu) * (d - mu))
    var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        cov = np.mean((d[lag:] - mu) * (d[:-lag] - mu))
        var += 2 * w * cov
    return var

def dm_test(loss_model: np.ndarray, loss_baseline: np.ndarray) -> Tuple[float, float]:
    d = loss_model - loss_baseline
    T = len(d)
    if T < 5:
        return float('nan'), float('nan')
    var = newey_west_var(d)
    if var <= 0 or math.isnan(var):
        return float('nan'), float('nan')
    t_stat = d.mean() / math.sqrt(var / T)
    if student_t is None:
        return t_stat, float('nan')
    p_value = 2 * student_t.sf(abs(t_stat), df=T-1)
    return t_stat, p_value

# ---------------------- Deterministische Normal-Quantile -----------------
def _norm_ppf(p: float) -> float:
    """Approximiere die Inverse der Standardnormal-CDF (ppf) deterministisch.
    Quelle: Peter John Acklam's rational approximation.
    """
    # Schutz und Clamping
    p = min(max(p, 1e-12), 1 - 1e-12)
    # Koeffizienten
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = math.sqrt(-2*math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

# ---------------------- Rolling Z-Score & Positional Encoding -----------
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism(True)
        except Exception:
            pass
    except Exception:
        pass

def _setup_run_logger(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("var_run")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    ch.setFormatter(fmt)
    log.addHandler(ch)
    # Leite warnings in Logger
    def _warn_to_log(message, category, filename, lineno, file=None, line=None):
        log.warning(warnings.formatwarning(message, category, filename, lineno))
    warnings.showwarning = _warn_to_log  # type: ignore
    return log

def rolling_zscore(df, window):
    mu = df.rolling(window, min_periods=window).mean()
    sigma = df.rolling(window, min_periods=window).std(ddof=0)
    # Epsilon-Floor gegen quasi-0-Varianz (stabil, realitätsnah):
    # σ_safe = max(σ, 1e-6)
    sigma_safe = sigma.clip(lower=1e-6)
    z = (df - mu) / sigma_safe
    # Entferne nur die anfänglichen Zeilen ohne vollständiges Fenster
    return z.iloc[window - 1:]

def realized_volatility(series: pd.Series, window: int = 21) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(pe, dtype=tf.float32)

# ---------------------- CLS-Token Layer ---------------------------------
class CLSToken(tf.keras.layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.d_model),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls, [batch_size, 1, 1])
        return tf.concat([cls_tokens, x], axis=1)

# MultiHeadAttention mit Zugriff auf letzte Attention-Scores
# ---------------------- Transformer-Block (Pre-LN, optional kausal) -----
def transformer_block(
    inputs,
    head_size=64,
    num_heads=4,
    ff_dim=128,
    dropout=0.2,
    causal=False,
    attention_dropout=0.1,
    layer_id: int = 0,
):
    # Pre-LayerNorm vor Self-Attention
    norm1 = LayerNormalization(epsilon=1e-6)(inputs)
    attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=attention_dropout,
        name=f"mha_{layer_id}",
    )(norm1, norm1, use_causal_mask=causal)
    attention = Dropout(dropout)(attention)
    x = Add()([inputs, attention])

    # Pre-LayerNorm vor Feed-Forward
    norm2 = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(ff_dim, activation="relu")(norm2)
    ff = Dropout(dropout)(ff)
    ff = Dense(x.shape[-1])(ff)
    x = Add()([x, ff])
    return x

# ---------------------- Transformer-Modell trainieren -------------------
def train_transformer_with_macro(
    df,
    feature_cols,
    q=0.05,
    seq_length=30,
    n_transformer_layers=4,
    head_size=16,
    num_heads=4,
    ff_dim=128,
    dropout=0.2,
    z_window=252,
    causal=False,
    attention_dropout=0.1,
    use_cls_token: bool = True,
    use_zscore: bool = True,
    use_returns_feature: bool = True,
    # Optimierung/Training
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 20,
    val_split: float = 0.2,
    clipnorm: float = 1.0,
    es_patience: int = 5,
    rlrop_patience: int = 3,
    rlrop_factor: float = 0.5,
):
    df_clean = df.copy()
    requested_features = list(dict.fromkeys(feature_cols))
    features = [f for f in requested_features if f in df_clean.columns]
    if not use_returns_feature:
        features = [f for f in features if f != 'returns']
    if len(features) == 0:
        raise ValueError("Keine der angeforderten Features im DataFrame gefunden.")
    # Sicherheitsfilter: entferne Features ohne ausreichende Historie vor Sequenz-/Z-Score-Bildung
    min_hist = seq_length + max(z_window, 20) + 5
    features_checked, features_dropped = filter_features_by_history(df_clean, features, min_hist)
    if len(features_checked) == 0:
        raise ValueError("Keine Features mit ausreichender Historie für dieses Fenster.")
    if len(features_dropped) > 0:
        logging.getLogger("var_run").info(f"Überspringe Features mit zu kurzer Historie: {features_dropped}")
    X_mat, target = prepare_design_matrix(df_clean, features_checked, z_window, use_zscore)
        

    if X_mat.shape[0] <= seq_length:
        raise ValueError(f"Nicht genügend Datenpunkte ({X_mat.shape[0]}) für Sequenzlänge {seq_length}.")

    X_scaled = X_mat.values.astype(np.float32)
    y_array = target.values.astype(np.float32)

    X, _ = create_sequences(X_scaled, seq_length)
    _, y = create_sequences(y_array.reshape(-1, 1), seq_length)
    X = X.reshape((X.shape[0], seq_length, X.shape[-1]))

    d_model = head_size * num_heads
    inputs = Input(shape=(seq_length, X.shape[-1]))
    x = Dense(d_model)(inputs)

    # Optional: CLS-Token vorschalten
    if use_cls_token:
        x = CLSToken(d_model)(x)
    pe_len = seq_length + (1 if use_cls_token else 0)
    pe_const = positional_encoding(pe_len, d_model)
    x = Lambda(lambda t, pe: t + pe, arguments={'pe': pe_const})(x)

    for li in range(n_transformer_layers):
        x = transformer_block(
            x,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            # CLS benötigt Zugriff auf die gesamte Sequenz; daher keine Kausalmaske
            causal=(False if use_cls_token else causal),
            attention_dropout=attention_dropout,
            layer_id=li,
        )

    # Optionale abschließende Norm und Head: CLS-Token oder Last-Token
    x = LayerNormalization(epsilon=1e-6)(x)
    if use_cls_token:
        x = Lambda(lambda t: t[:, 0, :])(x)
    else:
        x = Lambda(lambda t: t[:, -1, :])(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm), loss=quantile_loss(q))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=rlrop_factor, patience=rlrop_patience, min_lr=1e-6),
    ]
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=val_split,
        shuffle=False,
        callbacks=callbacks,
    )

    prep = {
        'features': features_checked,
        'z_window': z_window,
        'seq_length': seq_length,
        'use_cls_token': use_cls_token,
        'use_zscore': use_zscore,
        'use_returns_feature': use_returns_feature,
    }
    return model, prep, X, y

def permutation_importance_pinball(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, feature_names: list, q: float, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base_preds = model.predict(X, verbose=0).flatten()
    base_loss = pinball_loss_series(y, base_preds, q).mean()
    num_features = len(feature_names)
    degradations = []
    for f in range(num_features):
        losses = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # permutiere die Spalte f über alle Zeitsteps
            X_perm[:, :, f] = rng.permutation(X_perm[:, :, f])
            preds = model.predict(X_perm, verbose=0).flatten()
            losses.append(pinball_loss_series(y, preds, q).mean())
        degradations.append(np.mean(losses) - base_loss)
    return pd.DataFrame({'feature': feature_names, 'pinball_degradation': degradations}).sort_values('pinball_degradation', ascending=False)

# ---------------------- Evaluation & Visualisierung ---------------------
def evaluate_var_predictions(
    model,
    prep,
    df,
    feature_cols,
    q=0.05,
    label='Modell',
    baseline_preds: Optional[np.ndarray] = None,
    return_details: bool = False,
    save_plot_path: Optional[str] = None,
    do_plot: bool = True,
    verbose: bool = True,
):
    df_clean = df.copy()
    feats = [f for f in prep['features'] if f in list(dict.fromkeys(feature_cols)) and f in df_clean.columns]
    X_eval, target = prepare_design_matrix(df_clean, feats, prep['z_window'], prep.get('use_zscore', True))

    seq_length = prep['seq_length']
    X_pred, y_true, idx = build_sequences_from_df(X_eval, target, seq_length)

    preds = model.predict(X_pred).flatten()
    y_true = y_true.flatten()
    # idx kommt aus build_sequences_from_df

    breaches = y_true < preds
    breach_rate = float(np.mean(breaches))
    cvar = float(y_true[breaches].mean()) if np.any(breaches) else float('nan')
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    mae = float(np.mean(np.abs(y_true - preds)))
    pinball = float(pinball_loss_series(y_true, preds, q).mean())
    coverage_error = breach_rate - q
    # Tail-Metriken
    if np.any(breaches):
        tail_errors = y_true[breaches] - preds[breaches]
        tail_mae = float(np.mean(np.abs(tail_errors)))
        tail_rmse = float(np.sqrt(np.mean(tail_errors ** 2)))
    else:
        tail_mae = float('nan')
        tail_rmse = float('nan')
    # Brier-Score für Verletzungsindikator gegen nominales alpha
    brier = float(np.mean((breaches.astype(float) - q) ** 2))
    # Regime-abhängige Coverage (VIX hoch/niedrig), sauber auf idx reindizieren
    coverage_high = coverage_low = float('nan')
    if 'vix' in df_clean.columns:
        vix_aligned = df_clean['vix'].reindex(idx)
        vix_values = vix_aligned.values.astype(float)
        valid = ~np.isnan(vix_values)
        if valid.any():
            median_vix = np.nanmedian(vix_values)
            high_mask = (vix_values >= median_vix) & valid
            low_mask = (vix_values < median_vix) & valid
            if high_mask.any():
                coverage_high = float(np.mean(breaches[high_mask]))
            if low_mask.any():
                coverage_low = float(np.mean(breaches[low_mask]))

    # Backtests
    LR_uc, p_uc = kupiec_test(breaches, alpha=q)
    LR_ind, p_ind, LR_cc, p_cc = christoffersen_test(breaches, alpha=q)

    # DM-Test gegen Baseline (falls gegeben)
    dm_t, dm_p = (float('nan'), float('nan'))
    if baseline_preds is not None and len(baseline_preds) == len(preds):
        base_pinball = pinball_loss_series(y_true, baseline_preds, q)
        model_pinball = pinball_loss_series(y_true, preds, q)
        dm_t, dm_p = dm_test(model_pinball, base_pinball)

    df_results = pd.DataFrame({
        'Date': idx.values,
        'Return': y_true,
        'VaR': preds,
        'Breach': breaches
    })
    df_results.set_index('Date', inplace=True)

    if verbose:
        print(f"\n📊 {label} Ergebnisse:")
        print(
            f"Breach Rate: {breach_rate:.2%} (Err {coverage_error:+.2%}) | CVaR: {cvar:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Pinball: {pinball:.6f} | TailMAE: {tail_mae:.4f} | TailRMSE: {tail_rmse:.4f} | Brier: {brier:.6f}"
        )
        print(
            f"Kupiec LR_uc={LR_uc:.3f} (p={p_uc:.3f}) | Christoffersen IND LR={LR_ind:.3f} (p={p_ind:.3f}) | CC LR={LR_cc:.3f} (p={p_cc:.3f})"
        )
        if not math.isnan(dm_t):
            print(f"Diebold-Mariano vs. Baseline: t={dm_t:.3f}, p={dm_p:.3f}")

    if do_plot:
        plt.figure(figsize=(14, 6))
        # x-Achse mit Datum
        plt.plot(idx, y_true, label=f'{label}: Daily Returns')
        plt.plot(idx, preds, label=f'{label}: VaR {int((1-q)*100)}%', color='red')
        breach_dates = idx[breaches]
        plt.scatter(breach_dates, y_true[breaches], color='black', marker='x', label=f'{label}: Breach')
        plt.legend()
        plt.title(
            f"{label} – Transformer VaR {int((1-q)*100)}%\nBreach: {breach_rate:.2%} | CVaR: {cvar:.4f} | RMSE: {rmse:.4f}"
        )
        plt.grid(True)
        plt.xlabel('Datum')
        plt.ylabel('Rendite / VaR')
        plt.tight_layout()
        if save_plot_path is not None:
            try:
                Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_plot_path, dpi=150)
            except Exception as e:
                print(f"⚠️  Konnte Plot nicht speichern ({e})")
        plt.close()

    metrics = {
        'breach_rate': breach_rate,
        'coverage_error': coverage_error,
        'cvar': cvar,
        'rmse': rmse,
        'mae': mae,
        'pinball': pinball,
        'tail_mae': tail_mae,
        'tail_rmse': tail_rmse,
        'brier': brier,
        'coverage_high_vix': coverage_high,
        'coverage_low_vix': coverage_low,
        'kupiec_LR': LR_uc,
        'kupiec_p': p_uc,
        'christoffersen_ind_LR': LR_ind,
        'christoffersen_ind_p': p_ind,
        'christoffersen_cc_LR': LR_cc,
        'christoffersen_cc_p': p_cc,
        'dm_t': dm_t,
        'dm_p': dm_p,
    }
    if return_details:
        return df_results, metrics
    return df_results


# ---------------------- Walk-Forward Evaluation -------------------------
def walk_forward_evaluate(
    df,
    feature_cols,
    q=0.05,
    seq_length=30,
    n_transformer_layers: int = 4,
    z_window=252,
    head_size=16,
    num_heads=4,
    ff_dim=128,
    dropout=0.2,
    attention_dropout=0.1,
    use_cls_token: bool = True,
    use_zscore: bool = True,
    causal: bool = False,
    # Kalibrierung
    calib_mode: str = 'none',  # 'none' | 'scale' | 'conformal'
    calib_window: int = 252,
    calib_scale_min: float = 0.5,
    calib_scale_max: float = 2.0,
    # Refit-Steuerung
    refit_freq: str = 'Y',  # 'Y' jährlich, 'Q' quartalsweise, 'M' monatlich
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    # Rückwärtskompatibilität
    start_year: Optional[int] = 2021,
    end_year: Optional[int] = None,
    save_dir: Optional[str] = None,
    enable_plots: bool = True,
):
    df = df.sort_index()
    if start_date is None:
        if start_year is None:
            start_date = df.index.min()
        else:
            start_date = pd.Timestamp(f"{start_year}-01-01")
    if end_date is None:
        if end_year is None:
            end_date = df.index.max()
        else:
            end_date = pd.Timestamp(f"{end_year}-12-31")

    # Perioden gemäß gewünschter Refit-Frequenz
    periods = pd.period_range(start=start_date, end=end_date, freq=refit_freq)

    all_window_metrics = []
    all_pi_records = []  # für WF Feature-Dynamik (je Fenster)
    all_window_metrics_compare = []
    all_preds_transformer = []
    all_preds_garch = []
    all_preds_lstm = []
    all_returns = []
    for period in periods:
        window_start = period.start_time.normalize()
        window_end = period.end_time.normalize()
        train_df = df.loc[df.index < window_start]
        if train_df.shape[0] < (z_window + seq_length + 10):
            continue
        try:
            model, prep, _, _ = train_transformer_with_macro(
                train_df,
                feature_cols,
                q=q,
                seq_length=seq_length,
                n_transformer_layers=n_transformer_layers,
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                z_window=z_window,
                causal=causal,
                attention_dropout=attention_dropout,
                use_cls_token=use_cls_token,
                use_zscore=use_zscore,
                learning_rate=1e-3,
                batch_size=32,
                epochs=20,
                val_split=0.2,
                clipnorm=1.0,
                es_patience=5,
                rlrop_patience=3,
                rlrop_factor=0.5,
            )
        except Exception as e:
            logging.getLogger("var_run").error(f"Training-Fehler im Fenster {window_start.date()}–{window_end.date()}: {e}")
            continue
        # Volle Evaluation (ohne Plot) erzeugen, danach auf Fenster beschränken
        df_res, metrics = evaluate_var_predictions(
            model,
            prep,
            df,
            feature_cols,
            q=q,
            label=f"WF {str(period)}",
            return_details=True,
            save_plot_path=None,
            do_plot=False,
            verbose=False,
        )
        # Kalibrierungsfenster aus der Vergangenheit (vor window_start)
        def _compute_scale(preds_cal: np.ndarray, y_cal: np.ndarray, q_target: float) -> float:
            return compute_scale_from_grid(y_cal, preds_cal, q_target, calib_scale_min, calib_scale_max, steps=61)

        def _compute_conformal_shift(preds_cal: np.ndarray, y_cal: np.ndarray, q_target: float) -> float:
            return compute_conformal_shift(y_cal, preds_cal, q_target)

        # Filter auf das Prognosefenster
        df_res_year = df_res.loc[(df_res.index >= window_start) & (df_res.index <= window_end)]
        if df_res_year.empty:
            continue
        # Sammle Transformer-Preds & Returns (mit optionaler Kalibrierung)
        y_true = df_res_year['Return'].values
        preds = df_res_year['VaR'].values
        # Jahresverzeichnis für Outputs (Plots & CSV)
        yr_dir = Path(save_dir) / str(period) if save_dir is not None else None
        if yr_dir is not None:
            try:
                yr_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        # Rolling Coverage je Fenster speichern
        try:
            if yr_dir is not None and enable_plots:
                plot_rolling_coverage(df_res_year, q=q, window=63, save_path=yr_dir / f"wf_{str(period)}_rolling_coverage.png")
        except Exception as e:
            logging.getLogger("var_run").warning(f"Rolling Coverage Plot Problem {str(period)}: {e}")

        if calib_mode in ('scale', 'conformal'):
            df_cal = df_res.loc[df_res.index < window_start].tail(calib_window)
            if len(df_cal) >= max(30, seq_length + 5):
                y_cal = df_cal['Return'].values
                p_cal = df_cal['VaR'].values
                if calib_mode == 'scale':
                    s = _compute_scale(p_cal, y_cal, q)
                    preds = s * preds
                elif calib_mode == 'conformal':
                    shift = _compute_conformal_shift(p_cal, y_cal, q)
                    preds = preds + shift
            # Falls zu wenig Kalibrierungsdaten: keine Anpassung

        all_preds_transformer.append(preds)
        all_returns.append(y_true)

        # Baseline 1 (vereinfacht): Plain GARCH(1,1) mit Student‑t, tägliche 1‑Step‑Ahead‑Updates (zeitvariable VaR‑Kurve)
        try:
            ret_tr = train_df['returns'].dropna().astype(float) * 100.0
            n_oos = len(df_res_year.index)
            if len(ret_tr) > max(200, seq_length + z_window) and n_oos > 0:
                garch_model = arch_model(ret_tr, mean='zero', vol='GARCH', p=1, q=1, dist='t')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    garch_res = garch_model.fit(disp='off')

                # Parameter
                try:
                    params = garch_res.params
                    omega = float(params.get('omega', np.nan)) if hasattr(params, 'get') else float(params['omega'])
                    alpha1 = float(params['alpha[1]'])
                    beta1 = float(params['beta[1]'])
                    nu = float(params['nu']) if 'nu' in params.index else 8.0
                except Exception:
                    omega = float('nan'); alpha1 = float('nan'); beta1 = float('nan'); nu = 8.0

                if not (np.isfinite(omega) and np.isfinite(alpha1) and np.isfinite(beta1)):
                    raise RuntimeError('Ungültige GARCH‑Parameter')

                # Startwerte: letzte Trainings‑Sigma und Residuum
                try:
                    last_sigma = float(np.asarray(garch_res.conditional_volatility)[-1])
                    last_resid = float(np.asarray(garch_res.resid)[-1])
                except Exception:
                    last_sigma = float(np.std(ret_tr, ddof=0))
                    last_resid = float(ret_tr.iloc[-1])

                sigma2_t = max(last_sigma, 1e-6) ** 2
                eps_t = last_resid

                # t‑Quantil auf Varianz 1 normieren
                if student_t is not None and nu > 2:
                    z_alpha_std = float(student_t.ppf(q, df=nu) / math.sqrt(nu / (nu - 2)))
                else:
                    z_alpha_std = _norm_ppf(q)

                garch_var = np.empty(n_oos, dtype=float)
                # Reale OOS‑Returns in Prozent (für Residuen‑Update)
                y_oos_pct = (y_true * 100.0).astype(float)
                for i in range(n_oos):
                    # 1‑Step‑Ahead‑Varianz
                    sigma2_next = omega + alpha1 * (eps_t ** 2) + beta1 * sigma2_t
                    sigma_next = math.sqrt(max(sigma2_next, 1e-12))
                    # VaR in Returns‑Einheiten (nicht Prozent)
                    garch_var[i] = z_alpha_std * (sigma_next / 100.0)
                    # Nach Beobachtung von y_t: Update Zustände
                    eps_t = float(y_oos_pct[i])  # mean=0 -> Residuum = y_t
                    sigma2_t = sigma2_next
            else:
                garch_var = np.full(n_oos, np.nan)
        except Exception:
            garch_var = np.full(len(df_res_year), np.nan)
        all_preds_garch.append(garch_var)

        # Baseline 2: Multivariate LSTM-Quantil mit denselben Features/Preprocessing wie Transformer
        try:
            feats_mv = [f for f in prep['features'] if f in train_df.columns]
            X_tr, y_tr = prepare_design_matrix(train_df, feats_mv, z_window, use_zscore)
            X_l, y_l, _ = build_sequences_from_df(X_tr, y_tr, seq_length)
            if len(X_l) > 100:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM as K_LSTM
                lstm_model = Sequential([
                    K_LSTM(32, return_sequences=True, input_shape=(seq_length, X_l.shape[-1])),
                    Dropout(0.2),
                    K_LSTM(16),
                    Dropout(0.2),
                    Dense(1, activation=lambda t: -tf.nn.softplus(t))
                ])
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
                lstm_model.compile(optimizer=optimizer, loss=quantile_loss(q))
                es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
                lstm_model.fit(X_l, y_l, epochs=30, batch_size=64, verbose=0, shuffle=False, validation_split=0.1, callbacks=[es, rlrop])
                # Vorhersage über das gesamte Universum; dann exakt auf das Fenster mappen
                X_full_mv, y_full_mv = prepare_design_matrix(df, feats_mv, z_window, use_zscore)
                X_all, y_all, idx_all = build_sequences_from_df(X_full_mv, y_full_mv, seq_length)
                if len(X_all) > 0:
                    mask = (idx_all >= window_start) & (idx_all <= window_end)
                    X_oos_seq = X_all[mask]
                    idx_oos_seq = idx_all[mask]
                    if len(X_oos_seq) > 0:
                        lstm_preds = lstm_model.predict(X_oos_seq, verbose=0).flatten()
                        lstm_ser = pd.Series(lstm_preds, index=idx_oos_seq)
                        lstm_var = lstm_ser.reindex(df_res_year.index).values
                    else:
                        lstm_var = np.full(len(df_res_year), np.nan)
                else:
                    lstm_var = np.full(len(df_res_year), np.nan)
            else:
                lstm_var = np.full(len(df_res_year), np.nan)
        except Exception:
            lstm_var = np.full(len(df_res_year), np.nan)
        # Längenhygiene: Baselines exakt auf Fensterlänge trimmen
        if isinstance(garch_var, np.ndarray) and len(garch_var) != len(df_res_year.index):
            garch_var = pd.Series(garch_var, index=df_res_year.index[:len(garch_var)]).reindex(df_res_year.index).values
        if isinstance(lstm_var, np.ndarray) and len(lstm_var) != len(df_res_year.index):
            lstm_var = pd.Series(lstm_var, index=df_res_year.index[:len(lstm_var)]).reindex(df_res_year.index).values
        all_preds_lstm.append(lstm_var)
        # Historischer VaR vollständig entfernt
        hist_var = np.full(len(df_res_year), np.nan)
        # Transformer-Plot nur fürs Fenster
        try:
            if enable_plots:
                fig_t = plt.figure(figsize=(14, 6))
                plt.plot(df_res_year.index, y_true, label='Daily Returns', color='tab:blue', linewidth=1.2)
                b_mask = y_true < preds
                plt.plot(df_res_year.index, preds, label=f'Transformer VaR {int((1-q)*100)}%', color='red', linewidth=1.4)
                plt.scatter(df_res_year.index[b_mask], y_true[b_mask], color='black', marker='x', label='Breach')
                plt.title(f"Walk-Forward {str(period)} – Transformer VaR {int((1-q)*100)}%\nBreach: {float(np.mean(b_mask)):.2%} | CVaR: {float(y_true[b_mask].mean()) if b_mask.any() else float('nan'):.4f} | RMSE: {float(np.sqrt(np.mean((y_true - preds) ** 2))):.4f}")
                plt.xlabel('Datum')
                plt.ylabel('Rendite / VaR')
                plt.legend(); plt.grid(True); plt.tight_layout()
                target_path = (yr_dir / f"wf_{str(period)}.png") if yr_dir is not None else (plots_dir / f"wf_{str(period)}.png")
                save_figure(fig_t, target_path)
                plt.close(fig_t)
        except Exception as e:
            print(f"⚠️  Konnte WF-Plot nicht erzeugen ({e})")

        # Vergleichsplot: Returns vs. Transformer/GARCH/LSTM VaR
        try:
            if enable_plots:
                fig = plt.figure(figsize=(14, 6))
                n = len(y_true)
                plt.plot(df_res_year.index[:n], y_true[:n], label='Daily Returns', color='tab:blue', linewidth=1.2)
                plt.plot(df_res_year.index[:n], preds[:n], label=f'Transformer VaR {int((1-q)*100)}%', color='red', linewidth=1.4)
                if isinstance(garch_var, np.ndarray):
                    g = garch_var[:n]
                    if not np.all(np.isnan(g)):
                        plt.plot(df_res_year.index[:n], g, label=f'GARCH VaR {int((1-q)*100)}%', color='green', linestyle='--', linewidth=1.2)
                if isinstance(lstm_var, np.ndarray):
                    l = lstm_var[:n]
                    if not np.all(np.isnan(l)):
                        plt.plot(df_res_year.index[:n], l, label=f'LSTM VaR {int((1-q)*100)}%', color='orange', linestyle=':', linewidth=1.2)
                # Hist-VaR entfernt
                plt.title(f"Walk-Forward {str(period)} – Daily Returns vs. VaR (Transformer/GARCH/LSTM)")
                plt.xlabel('Datum')
                plt.ylabel('Rendite / VaR')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                target_path = (yr_dir / f"wf_{str(period)}_compare.png") if yr_dir is not None else (plots_dir / f"wf_{str(period)}_compare.png")
                save_figure(fig, target_path)
                plt.close(fig)
                # Rolling Coverage Compare (Transformer/GARCH/LSTM/Hist) pro Jahr
                try:
                    if enable_plots:
                        rc_df = df_res_year.copy()
                        if isinstance(garch_var, np.ndarray):
                            rc_df['GARCH_VaR'] = garch_var[:n]
                        if isinstance(lstm_var, np.ndarray):
                            rc_df['LSTM_VaR'] = lstm_var[:n]
                        if isinstance(hist_var, np.ndarray):
                            rc_df['HIST_VaR'] = hist_var[:n]
                        if yr_dir is not None:
                            plot_rolling_coverage_compare(rc_df, q=q, window=63, save_path=yr_dir / f"wf_{str(period)}_rolling_coverage_compare.png")
                except Exception as e:
                    logging.getLogger("var_run").warning(f"WF Rolling Coverage Compare Problem {str(period)}: {e}")
        except Exception as e:
            print(f"⚠️  Konnte Vergleichsplot nicht erzeugen ({e})")

        # Serien-CSV fürs Fenster speichern (Transformer + Baselines)
        try:
            if save_dir is not None:
                out_df = pd.DataFrame({
                    'Return': y_true,
                    'VaR': preds,
                    'Breach': y_true < preds,
                    'GARCH_VaR': garch_var,
                    'LSTM_VaR': lstm_var,
                }, index=df_res_year.index)
                # Per-Jahres-Unterordner für WF-Ergebnisse
                yr_dir = Path(save_dir) / str(period)
                yr_dir.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(yr_dir / f"wf_{str(period)}_series.csv")
        except Exception as e:
            logging.getLogger("var_run").warning(f"Serien-CSV Problem {str(period)}: {e}")

        # Z-Scores für Training und OOS-Fenster speichern
        try:
            if save_dir is not None and prep.get('use_zscore', True):
                feats = [f for f in prep['features'] if f in df.columns]
                if len(feats) > 0:
                    # Training-Z: Historie vor window_start
                    hist_start = max(df.index.min(), window_start - pd.Timedelta(days=int(z_window*2)))
                    train_slice = df.loc[(df.index >= hist_start) & (df.index < window_start), feats].ffill()
                    z_tr = rolling_zscore(train_slice, window=z_window).replace([np.inf, -np.inf], np.nan)
                    yr_dir = Path(save_dir) / str(period)
                    yr_dir.mkdir(parents=True, exist_ok=True)
                    z_tr.to_csv(yr_dir / f"wf_{str(period)}_zscore_train.csv")
                    # OOS-Z: auf OOS-Index
                    inputs_full = df[feats].ffill()
                    z_full = rolling_zscore(inputs_full, window=z_window).replace([np.inf, -np.inf], np.nan)
                    z_oos = z_full.reindex(df_res_year.index)
                    yr_dir = Path(save_dir) / str(period)
                    yr_dir.mkdir(parents=True, exist_ok=True)
                    z_oos.to_csv(yr_dir / f"wf_{str(period)}_zscore_oos.csv")
        except Exception as e:
            logging.getLogger("var_run").warning(f"Z-Score Export Problem {str(period)}: {e}")
        # WF-Fenster: Permutation Importance berechnen (optional gedrosselt)
        try:
            feats_in_model = [f for f in prep['features'] if f in df.columns]
            if len(feats_in_model) > 0:
                X_full_pi, y_full_pi = prepare_design_matrix(df, feats_in_model, z_window, use_zscore)
                X_pi, y_pi, idx_pi = build_sequences_from_df(X_full_pi, y_full_pi, seq_length)
                mask_pi = (idx_pi >= window_start) & (idx_pi <= window_end)
                X_pi_win = X_pi[mask_pi]; y_pi_win = y_pi[mask_pi]
                if len(X_pi_win) > 0:
                    sample = min(512, len(X_pi_win))
                    pi_df = permutation_importance_pinball(
                        model, X_pi_win[:sample], y_pi_win[:sample], feats_in_model, q=q, n_repeats=3
                    )
                    if yr_dir is not None:
                        pi_df.to_csv(yr_dir / f"wf_{str(period)}_permutation_importance.csv", index=False)
                        # Per-Period Permutation Importance (Top 20) – Balkendiagramm
                        try:
                            if enable_plots:
                                top = pi_df.sort_values('pinball_degradation', ascending=False).head(20).iloc[::-1]
                                fig_pi = plt.figure(figsize=(10, 6))
                                plt.barh(top['feature'].astype(str), top['pinball_degradation'].astype(float), color='tab:gray')
                                plt.xlabel('Pinball-Degradation (Δ-Loss)'); plt.ylabel('Feature')
                                plt.title(f'Permutation Importance – Top 20 ({str(period)})')
                                plt.tight_layout(); fig_pi.savefig(yr_dir / f"wf_{str(period)}_permutation_importance.png", dpi=150)
                                plt.close(fig_pi)
                        except Exception:
                            pass
                    pi_df['period'] = str(period)
                    all_pi_records.append(pi_df)
        except Exception as e:
            logging.getLogger("var_run").warning(f"WF PI Problem {str(period)}: {e}")
        breaches = y_true < preds
        breach_rate = float(breaches.mean())
        cvar = float(y_true[breaches].mean()) if breaches.any() else float('nan')
        rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
        mae = float(np.mean(np.abs(y_true - preds)))
        pinball = float(pinball_loss_series(y_true, preds, q).mean())
        brier = float(np.mean((breaches.astype(float) - q) ** 2))
        LR_uc, p_uc = kupiec_test(breaches, alpha=q)
        LR_ind, p_ind, LR_cc, p_cc = christoffersen_test(breaches, alpha=q)
        rec_t = {
            'period': str(period),
            'breach_rate': breach_rate,
            'coverage_error': breach_rate - q,
            'cvar': cvar,
            'rmse': rmse,
            'mae': mae,
            'pinball': pinball,
            'brier': brier,
            'kupiec_LR': LR_uc,
            'kupiec_p': p_uc,
            'christoffersen_ind_LR': LR_ind,
            'christoffersen_ind_p': p_ind,
            'christoffersen_cc_LR': LR_cc,
            'christoffersen_cc_p': p_cc,
        }
        all_window_metrics.append(rec_t)
        rt_cmp = rec_t.copy(); rt_cmp['model'] = 'Transformer'
        all_window_metrics_compare.append(rt_cmp)

        # Compare: GARCH
        if isinstance(garch_var, np.ndarray) and not np.all(np.isnan(garch_var)):
            add_compare_metrics(all_window_metrics_compare, str(period), 'GARCH', y_true, garch_var, q)

        # Compare: LSTM
        if isinstance(lstm_var, np.ndarray) and not np.all(np.isnan(lstm_var)):
            add_compare_metrics(all_window_metrics_compare, str(period), 'LSTM', y_true, lstm_var, q)
        # Compare: HIST (rolling quantile)
        if isinstance(hist_var, np.ndarray) and not np.all(np.isnan(hist_var)):
            add_compare_metrics(all_window_metrics_compare, str(period), 'HIST', y_true, hist_var, q)
        # Per-Period Compare-Metriken und -Plots im Jahresordner speichern
        try:
            if save_dir is not None:
                cdf_p = pd.DataFrame(all_window_metrics_compare)
                cdf_p = cdf_p[cdf_p['period'] == str(period)]
                if not cdf_p.empty:
                    yr_dir_local = Path(save_dir) / str(period)
                    yr_dir_local.mkdir(parents=True, exist_ok=True)
                    cdf_p.to_csv(yr_dir_local / f"wf_{str(period)}_metrics_compare.csv", index=False)
                    if enable_plots:
                        # Per-Period Heatmap
                        hm_cols = ['breach_rate','coverage_error','cvar','rmse','mae','pinball','brier','kupiec_p','christoffersen_cc_p']
                        fig, ax = plt.subplots(figsize=(12, 4))
                        cdf_p['row'] = cdf_p['model']
                        data = cdf_p.set_index('row')[hm_cols].astype(float)
                        im = ax.imshow(data.values, aspect='auto', cmap='viridis')
                        ax.set_yticks(range(len(data.index))); ax.set_yticklabels(data.index)
                        ax.set_xticks(range(len(hm_cols)))
                        ax.set_xticklabels(['Breach','CovErr','CVaR','RMSE','MAE','Pinball','Brier','Kupiec p','CC p'], rotation=45, ha='right')
                        ax.set_title(f'Metriken-Heatmap (Periode {str(period)} × Modell × Kennzahl)')
                        fig.colorbar(im, ax=ax, label='Skalierter Wert')
                        plt.tight_layout(); plt.savefig(yr_dir_local / f"wf_{str(period)}_metrics_heatmap_compare.png", dpi=150); plt.close(fig)
                        # Per-Period Balkendiagramme
                        fig, axes = plt.subplots(3, 3, figsize=(16, 10))
                        for a, col in zip(axes.ravel(), hm_cols):
                            pivot = cdf_p.pivot(index='period', columns='model', values=col)
                            pivot.plot(kind='bar', ax=a, width=0.8)
                            a.set_title(col); a.set_xlabel('Periode'); a.set_ylabel(col); a.grid(True, axis='y', alpha=0.3)
                        plt.tight_layout(); plt.savefig(yr_dir_local / f"wf_{str(period)}_metrics_bars_compare.png", dpi=150); plt.close(fig)
        except Exception as e:
            logging.getLogger("var_run").warning(f"Per-Period Compare-Export Problem {str(period)}: {e}")
    # Optionale gesammelte Serien zurückgeben (für DM-Tests über das ganze Intervall)
    results_df = pd.DataFrame(all_window_metrics)
    # WF-Feature-Dynamik Heatmap über Fenster speichern
    try:
        if save_dir is not None and len(all_pi_records) > 0:
            pi_all = pd.concat(all_pi_records, ignore_index=True)
            pi_all.to_csv(Path(save_dir) / 'wf_permutation_importance.csv', index=False)
            if enable_plots:
                # Heatmap: Perioden × Feature (Top 20 im Mittel)
                top_feats = (
                    pi_all.groupby('feature')['pinball_degradation']
                    .mean()
                    .sort_values(ascending=False)
                    .head(20)
                    .index.astype(str)
                    .tolist()
                )
                sub = pi_all[pi_all['feature'].astype(str).isin(top_feats)].copy()
                piv = sub.pivot_table(index='period', columns='feature', values='pinball_degradation', aggfunc='mean').fillna(0.0)
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(piv.values, aspect='auto', cmap='viridis')
                ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
                ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([str(c) for c in piv.columns], rotation=45, ha='right')
                ax.set_title('WF Feature-Dynamik – Permutation Importance (Top 20)')
                fig.colorbar(im, ax=ax, label='Pinball-Degradation')
                # Speichern im plots/ Unterordner des Runs
                try:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                plt.tight_layout(); plt.savefig(Path(save_dir) / 'plots' / 'wf_feature_dynamics_heatmap.png', dpi=150); plt.close(fig)
    except Exception as e:
        logging.getLogger("var_run").warning(f"WF Feature-Dynamik Heatmap Problem: {e}")
    # Export Compare-Metriken, falls save_dir gesetzt
    try:
        if save_dir is not None and len(all_window_metrics_compare) > 0:
            pd.DataFrame(all_window_metrics_compare).to_csv(Path(save_dir) / 'wf_metrics_compare.csv', index=False)
            # Zusätzliche Vergleichs-Heatmap direkt erzeugen
            try:
                cdf = pd.DataFrame(all_window_metrics_compare)
                if not cdf.empty and enable_plots:
                    hm_cols = ['breach_rate','coverage_error','cvar','rmse','mae','pinball','brier','kupiec_p','christoffersen_cc_p']
                    fig, ax = plt.subplots(figsize=(16, 8))
                    cdf['row'] = cdf['period'].astype(str) + ' – ' + cdf['model']
                    data = cdf.set_index('row')[hm_cols].astype(float)
                    im = ax.imshow(data.values, aspect='auto', cmap='viridis')
                    ax.set_yticks(range(len(data.index))); ax.set_yticklabels(data.index)
                    ax.set_xticks(range(len(hm_cols)))
                    ax.set_xticklabels(['Breach','CovErr','CVaR','RMSE','MAE','Pinball','Brier','Kupiec p','CC p'], rotation=45, ha='right')
                    ax.set_title('Metriken-Heatmap (Periode × Modell × Kennzahl)')
                    fig.colorbar(im, ax=ax, label='Skalierter Wert')
                    plt.tight_layout(); plt.savefig(Path(save_dir) / 'plots' / 'metrics_heatmap_compare.png', dpi=150); plt.close(fig)
            except Exception:
                pass
    except Exception as e:
        logging.getLogger("var_run").warning(f"Compare-Metriken Export Problem: {e}")
    if len(all_preds_transformer) > 0:
        return results_df, {
            'transformer_var': np.concatenate(all_preds_transformer),
            'garch_var': np.concatenate(all_preds_garch) if len(all_preds_garch) == len(all_preds_transformer) else None,
            'lstm_var': np.concatenate(all_preds_lstm) if len(all_preds_lstm) == len(all_preds_transformer) else None,
            'returns': np.concatenate(all_returns),
        }
    return results_df, None


# ---------------------- Diagnose & Visualisierung -----------------------
def plot_transformer_diagnostics(df_results: pd.DataFrame, df_full: pd.DataFrame, q: float = 0.05, vol_window: int = 21, save_path: Optional[str] = None, include_baselines: bool = False):
    idx = df_results.index
    preds = df_results['VaR'].values
    returns = df_results['Return'].values
    rv = realized_volatility(df_full['returns'], window=vol_window).reindex(idx)
    # Optional: Baselines einbetten, falls vorhanden
    garch = df_results['GARCH_VaR'].reindex(idx).values if 'GARCH_VaR' in df_results.columns else None
    lstm = df_results['LSTM_VaR'].reindex(idx).values if 'LSTM_VaR' in df_results.columns else None

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    ax = axes[0, 0]
    ax.plot(idx, returns, label='Daily Returns', color='tab:blue', linewidth=1.2, alpha=0.8)
    ax.plot(idx, preds, label=f'Transformer VaR {int((1-q)*100)}%', color='red', linewidth=1.4)
    if include_baselines and garch is not None and not np.all(np.isnan(garch)):
        ax.plot(idx, garch, label=f'GARCH VaR {int((1-q)*100)}%', color='green', linestyle='--', linewidth=1.2)
    if include_baselines and lstm is not None and not np.all(np.isnan(lstm)):
        ax.plot(idx, lstm, label=f'LSTM VaR {int((1-q)*100)}%', color='orange', linestyle=':', linewidth=1.2)
    # Stelle sicher, dass keine künstliche Verlängerung über die Daten hinaus gezeichnet wird
    ax.set_xlim(idx.min(), idx.max())
    ax.set_title('Renditen vs. VaR')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Rendite / VaR')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    def _norm(x):
        x = pd.Series(x, index=idx)
        mu, sd = np.nanmean(x), np.nanstd(x)
        return (x - mu) / (sd if sd > 0 else 1.0)
    ax.plot(idx, _norm(preds), label='Transformer VaR (z-norm)', color='red', linewidth=1.2)
    if rv is not None:
        ax.plot(idx, _norm(rv.values), label=f'RV{vol_window} (z-norm)', color='tab:blue', linewidth=1.2)
    if include_baselines and garch is not None and not np.all(np.isnan(garch)):
        ax.plot(idx, _norm(garch), label='GARCH VaR (z-norm)', color='green', linestyle='--', linewidth=1.2)
    if include_baselines and lstm is not None and not np.all(np.isnan(lstm)):
        ax.plot(idx, _norm(lstm), label='LSTM VaR (z-norm)', color='orange', linestyle=':', linewidth=1.2)
    ax.set_title('Skalenvergleich (z-normalisiert)')
    ax.set_xlabel('Datum')
    ax.set_ylabel('z-Score')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    preds_s = pd.Series(preds, index=idx)
    roll_std = preds_s.rolling(63, min_periods=20).std()
    ax.plot(idx, roll_std, color='purple')
    ax.set_title('Rollende Std der VaR-Vorhersage (63 Tage)')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Standardabweichung')
    ax.grid(True)

    # unteres rechtes Feld leer halten
    axes[1, 1].axis('off')

    plt.tight_layout()
    if save_path is not None:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
        except Exception as e:
            print(f"⚠️  Konnte Diagnostics-Plot nicht speichern ({e})")
    else:
        plt.show()
    plt.close()


def plot_macro_overview(df: pd.DataFrame, macro_cols: Optional[list] = None, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None):
    df = df.sort_index()
    if start is not None or end is not None:
        df = df.loc[start:end]
    if macro_cols is None:
        macro_cols = [
            'cpi_yoy', 'unemployment_rate', 'housing_starts', 'nonfarm_payrolls',
            'consumer_confidence', 'fed_funds', 'epu_index', 't10y2y_spread', 't10y3m_spread',
            'hy_ccc_spread', 'corp_bond_spread'
        ]
    cols = [c for c in macro_cols if c in df.columns]
    if len(cols) == 0:
        warnings.warn('Keine Makro-Variablen zum Plotten gefunden.')
        return
    data = df[cols].copy()
    data = (data - data.mean()) / data.std(ddof=0)
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), squeeze=False)
    for i, c in enumerate(cols):
        r, k = divmod(i, ncols)
        ax = axes[r, k]
        ax.plot(data.index, data[c], label=c)
        ax.set_title(c)
        ax.grid(True)
        ax.legend()
    for j in range(n, nrows * ncols):
        r, k = divmod(j, ncols)
        axes[r, k].axis('off')
    plt.tight_layout()
    plt.show()


def plot_macro_overview_zscore_and_raw(df: pd.DataFrame, macro_cols: Optional[list] = None, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None, z_window: int = 252):
    df = df.sort_index()
    if start is not None or end is not None:
        df = df.loc[start:end]
    if macro_cols is None:
        macro_cols = [
            'cpi_yoy', 'unemployment_rate', 'housing_starts', 'nonfarm_payrolls',
            'consumer_confidence', 'fed_funds', 'epu_index', 't10y2y_spread', 't10y3m_spread',
            'hy_ccc_spread', 'corp_bond_spread'
        ]
    cols = [c for c in macro_cols if c in df.columns]
    if len(cols) == 0:
        warnings.warn('Keine Makro-Variablen zum Plotten gefunden.')
        return
    raw = df[cols].copy()
    z = rolling_zscore(raw, window=z_window)
    common_index = raw.index.intersection(z.index)
    raw = raw.loc[common_index]
    z = z.loc[common_index]

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.2 * n), squeeze=False)
    for i, c in enumerate(cols):
        ax = axes[i, 0]
        ax.plot(raw.index, raw[c], label=f'{c} (raw)', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(z.index, z[c], label=f'{c} (z)', color='tab:orange', alpha=0.6)
        ax.set_title(c)
        ax.grid(True)
    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

# ---------------------- Rolling Coverage Plot ----------------------------
def plot_rolling_coverage(df_results: pd.DataFrame, q: float = 0.05, window: int = 63, save_path: Optional[Path] = None):
    idx = df_results.index
    y = df_results['Return'].values
    p = df_results['VaR'].values
    breaches = (y < p).astype(float)
    roll_cov = pd.Series(breaches, index=idx).rolling(window, min_periods=max(5, window//3)).mean()
    plt.figure(figsize=(14, 4))
    plt.plot(idx, roll_cov, label=f'Rolling Coverage ({window}d)')
    plt.axhline(q, color='red', linestyle='--', label=f'Target q={q:.2f}')
    plt.title('Rolling Coverage vs. Target')
    plt.xlabel('Datum'); plt.ylabel('Coverage')
    plt.legend(); plt.grid(True); plt.tight_layout()
    if save_path is not None:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
        except Exception as e:
            print(f"⚠️  Konnte Rolling Coverage Plot nicht speichern ({e})")
    plt.close()

# ---------------- Rolling Coverage Compare (Transformer/GARCH/LSTM/HIST) ---
def plot_rolling_coverage_compare(df_results: pd.DataFrame, q: float = 0.05, window: int = 63, save_path: Optional[Path] = None):
    idx = df_results.index
    y = df_results['Return'].values
    series = {
        'Transformer': ('VaR', 'red', '-'),
        'GARCH': ('GARCH_VaR', 'green', '--'),
        'LSTM': ('LSTM_VaR', 'orange', ':'),
    }
    plt.figure(figsize=(14, 4))
    plotted = False
    for name, (col, color, style) in series.items():
        if col in df_results.columns and not df_results[col].isna().all():
            p = df_results[col].values
            breaches = (y < p).astype(float)
            roll_cov = pd.Series(breaches, index=idx).rolling(window, min_periods=max(5, window//3)).mean()
            plt.plot(idx, roll_cov, label=f'{name} Rolling Coverage ({window}d)', color=color, linestyle=style)
            plotted = True
    if not plotted:
        return
    plt.axhline(q, color='red', linestyle='--', label=f'Target q={q:.2f}')
    plt.title('Rolling Coverage vs. Target – Compare')
    plt.xlabel('Datum'); plt.ylabel('Coverage')
    plt.legend(); plt.grid(True); plt.tight_layout()
    if save_path is not None:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
        except Exception as e:
            print(f"⚠️  Konnte Rolling Coverage Compare Plot nicht speichern ({e})")
    plt.close()

# ---------------------- WF Feature-Dynamik Heatmap -----------------------
def plot_wf_feature_dynamics_heatmap(run_dir: Path, top_k: int = 15):
    """Aggregiert Permutation-Importance im Diagnostics-Run und visualisiert Top-k Features.
    Erwartet run_dir/plots/permutation_importance.csv; sonst kein Plot."""
    try:
        # Prefer Diagnostics OSS location, fallback to legacy root
        perm_csv = run_dir / 'OSS (20)' / 'permutation_importance.csv'
        if not perm_csv.exists():
            perm_csv = run_dir / 'permutation_importance.csv'
        if not perm_csv.exists():
            return
        pdf = pd.read_csv(perm_csv)
        top = pdf.sort_values('pinball_degradation', ascending=False).head(top_k)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top['feature'].astype(str).iloc[::-1], top['pinball_degradation'].astype(float).iloc[::-1], color='tab:gray')
        ax.set_title(f'Permutation Importance – Top {top_k}')
        ax.set_xlabel('Pinball-Degradation (Δ-Loss)'); ax.set_ylabel('Feature')
        plt.tight_layout(); fig.savefig(run_dir / 'plots' / 'permutation_importance_topk.png', dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"⚠️  Feature-Dynamik Plot fehlgeschlagen ({e})")

# ---------------------- Sensitivitäts-Grid Visualisierung ---------------
def plot_sensitivity_grid(*args, **kwargs):
    pass

# ------------------------ Ausführung -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=1987)
    parser.add_argument('--refit', type=str, default='Y', choices=['Y', 'Q', 'M'])
    parser.add_argument('--seq-length', type=int, default=60)
    # Transformer-Architektur
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--head-size', type=int, default=16)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--ff-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn-dropout', type=float, default=0.1)
    parser.add_argument('--causal', action='store_true')
    # Optimierung/Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--clipnorm', type=float, default=1.0)
    parser.add_argument('--es-patience', type=int, default=5)
    parser.add_argument('--rlrop-patience', type=int, default=3)
    parser.add_argument('--rlrop-factor', type=float, default=0.5)
    parser.add_argument('--zscore', action='store_true')
    parser.add_argument('--no-zscore', dest='zscore', action='store_false')
    parser.set_defaults(zscore=True)
    # Einheits-Flag für Returns-Feature
    parser.add_argument('--use-returns', dest='use_returns', action='store_true')
    parser.add_argument('--no-returns', dest='use_returns', action='store_false')
    parser.set_defaults(use_returns=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--z-window', type=int, default=126)
    parser.add_argument('--outdir', type=str, default='outputs')
    parser.add_argument('--oos-years', type=int, nargs='*', default=None, help='Spezifische OOS-Jahre (z. B. 2008 2009 2022 2025)')
    parser.add_argument('--diagnostics', action='store_true', help='Optionaler Diagnose-Run (Plots & Permutation Importance)')
    parser.add_argument('--filter-permutation', action='store_true', help='Makro-Features anhand letzter Permutationsimportance filtern (<=0 wird entfernt)')
    parser.add_argument('--timeframe', type=str, default='daily', choices=['daily', 'weekly', 'monthly'], help='Zeitauflösung für Training/Evaluation')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='Enddatum für Daten (YYYY-MM-DD)')
    # Kalibrierung per Refit-Fenster
    parser.add_argument('--calibration', type=str, default='none', choices=['none', 'scale', 'conformal'], help='Post-hoc Kalibrierung pro Refit-Fenster')
    parser.add_argument('--calib-window', type=int, default=252, help='Kalibrierungsfenster in Tagen (historische Länge)')
    parser.add_argument('--calib-scale-min', type=float, default=0.5, help='Minimale Skala für grid search')
    parser.add_argument('--calib-scale-max', type=float, default=2.0, help='Maximale Skala für grid search')
    # Plot-Steuerung
    parser.add_argument('--plots', dest='plots', action='store_true', help='Plots erzeugen und speichern')
    parser.add_argument('--no-plots', dest='plots', action='store_false', help='Keine Plots erzeugen')
    parser.set_defaults(plots=False)
    
    # Head-Auswahl: CLS-Head (default) oder Last-Token
    parser.add_argument('--cls-head', dest='use_cls_token', action='store_true', help='Verwende CLS-Token als Regressions-Head (Standard)')
    parser.add_argument('--last-token', dest='use_cls_token', action='store_false', help='Verwende letztes Sequenz-Token als Regressions-Head')
    parser.set_defaults(use_cls_token=True)
    # Optionale Features
    parser.add_argument('--use-spx-close', dest='use_spx_close', action='store_true', help='SPX Close als Feature verwenden')
    parser.add_argument('--no-spx-close', dest='use_spx_close', action='store_false', help='SPX Close nicht als Feature verwenden')
    parser.set_defaults(use_spx_close=False)
    # Datenabruf-Startjahr (für FRED/Yahoo)
    parser.add_argument('--data-start-year', type=int, default=1950, help='Startjahr für den Datenabruf (z. B. 1960)')
    # Cache-Steuerung
    parser.add_argument('--no-cache', dest='use_cache', action='store_false', help='Kein Cache verwenden')
    parser.add_argument('--refresh-cache', action='store_true', help='Cache ignorieren und neu laden')
    parser.set_defaults(use_cache=True, refresh_cache=False)
    args = parser.parse_args()

    # Lauf-spezifisches Verzeichnis erzeugen (Parameter-Fingerprint)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args)
    args_str = json.dumps(args_dict, sort_keys=True)
    run_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]
    run_dir = outdir_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_hash}"
    plots_dir = run_dir / 'plots'
    oss_dir = run_dir / 'OSS (20)'
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Erzeuge OSS (20) Verzeichnis nur, wenn Diagnostics aktiviert ist
    if args.diagnostics:
        try:
            oss_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    logger = _setup_run_logger(run_dir)
    # Parameter speichern
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)

    set_global_seed(args.seed)

    daily_df, weekly_df, monthly_df = prepare_macro_and_market_data(
        start=f"{args.data_start_year}-01-01",
        end=args.end_date,
        release_aware=True,
        cache_dir='data',
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
        fred_api_key=os.environ.get('FRED_API_KEY'),
    )
    # Wähle Master-DataFrame entsprechend timeframe
    if args.timeframe == 'weekly':
        df_master = weekly_df
    elif args.timeframe == 'monthly':
        df_master = monthly_df
    else:
        df_master = daily_df
    # Features aus den neuen FRED-Keys ableiten (+ abgeleitete Spreads) und Marktfeatures ergänzen
    fred_like = [
        'cpi_mom', 'pce_price_index', 'unemployment_rate', 'nonfarm_payrolls', 'industrial_production',
        'capacity_utilization', 'housing_starts', 'building_permits', 'retail_sales', 'durable_orders',
        'consumer_confidence', 'fed_funds', 'epu_index', 't10y2y_spread', 't10y3m_spread', 'hy_ccc_spread'
    ]
    macro_cols = [c for c in fred_like if c in df_master.columns]
    derived_cols = [c for c in ['corp_bond_spread'] if c in df_master.columns]
    base_feats = []
    if args.use_returns:
        base_feats.append('returns')
    if args.use_spx_close:
        base_feats.append('spx_close')
    # Optional: Makros anhand letzter Permutationsimportance filtern (<=0 entfernen)
    if args.filter_permutation:
        try:
            # Nur primär: verwende ausschließlich OSS (20)/permutation_importance.csv aus dem aktuellen Run
            perm_path = run_dir / 'OSS (20)' / 'permutation_importance.csv'
            if perm_path.exists():
                perm_df = pd.read_csv(perm_path)
                keep = set(perm_df.loc[perm_df['pinball_degradation'] > 0, 'feature'].astype(str))
                # Nur Makros/abgeleitete Makros filtern; Marktfeatures unangetastet
                dropped_macros = [c for c in (macro_cols + derived_cols) if c not in keep]
                kept_macros = [c for c in macro_cols if c in keep]
                kept_derived = [c for c in derived_cols if c in keep]
                if len(kept_macros) + len(kept_derived) == 0:
                    print("⚠️  Keine Makros mit positiver Permutationsimportance gefunden – verwende ungefilterte Makros.")
                else:
                    if len(dropped_macros) > 0:
                        print(f"🧹 Entferne Makros ohne positive Importance: {dropped_macros}")
                    macro_cols = kept_macros
                    derived_cols = kept_derived
            else:
                print("ℹ️  Keine permutation_importance.csv im Ordner OSS (20) gefunden – Filter wird übersprungen.")
        except Exception as e:
            print(f"⚠️  Konnte Permutationsimportance nicht auswerten ({e}) – Filter wird übersprungen.")
    feature_cols = base_feats + macro_cols + derived_cols

    logger.info("📆 Walk-Forward Evaluation...")
    if args.oos_years:
        frames = []
        for yr in args.oos_years:
            wf_y, _ = walk_forward_evaluate(
                df_master,
                feature_cols,
                q=0.05,
                seq_length=args.seq_length,
                n_transformer_layers=args.layers,
                z_window=args.z_window,
                use_zscore=args.zscore,
                head_size=args.head_size,
                num_heads=args.num_heads,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                attention_dropout=args.attn_dropout,
                use_cls_token=args.use_cls_token,
                causal=args.causal,
                calib_mode=args.calibration,
                calib_window=args.calib_window,
                calib_scale_min=args.calib_scale_min,
                calib_scale_max=args.calib_scale_max,
                refit_freq='Y',
                start_year=yr,
                end_year=yr,
                save_dir=str(run_dir),
                enable_plots=args.plots,
            )
            frames.append(wf_y)
        wf_metrics = pd.concat(frames, ignore_index=True)
        series = None
    else:
        wf_metrics, series = walk_forward_evaluate(
            df_master,
            feature_cols,
            q=0.05,
            seq_length=args.seq_length,
            n_transformer_layers=args.layers,
            z_window=args.z_window,
            use_zscore=args.zscore,
            head_size=args.head_size,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            attention_dropout=args.attn_dropout,
            use_cls_token=args.use_cls_token,
            causal=args.causal,
            calib_mode=args.calibration,
            calib_window=args.calib_window,
            calib_scale_min=args.calib_scale_min,
            calib_scale_max=args.calib_scale_max,
            refit_freq=args.refit,
            start_year=args.start_year,
            save_dir=str(run_dir),
            enable_plots=args.plots,
        )
    print(wf_metrics)
    wf_metrics.to_csv(run_dir / 'wf_metrics.csv', index=False)
    # Sensitivity-Grid entfernt; Auswertung erfolgt über runs_dashboard.py

    # Optionaler Diagnose-Run (nur auf Wunsch)
    if args.diagnostics:
        # Chronologische 80/20-Splits für Diagnose
        idx = df_master.index.sort_values()
        split_i = int(len(idx) * 0.8)
        split_date = idx[split_i]
        train_df = df_master.loc[df_master.index <= split_date]
        eval_df = df_master.loc[df_master.index > split_date]
        # Safety: genügend Daten für Sequenzen und Z-Score?
        min_needed = args.seq_length + max(args.z_window, 20) + 5
        if len(train_df.index) < min_needed or len(eval_df.index) < (args.seq_length + 5):
            print("⚠️  Überspringe Diagnostics: zu wenige Datenpunkte nach Split/Sequenzen.")
        else:
            model, prep, X_train_seq, y_train_seq = train_transformer_with_macro(
                train_df,
                feature_cols,
                q=0.05,
                seq_length=args.seq_length,
                z_window=args.z_window,
                head_size=16,
                num_heads=4,
                ff_dim=128,
                dropout=0.2,
                attention_dropout=0.1,
                use_cls_token=args.use_cls_token,
                use_zscore=args.zscore,
                use_returns_feature=args.use_returns,
            )
            # Evaluate auf OOS 20%-Fenster (unkalibriert)
            # Transformer-only Series-Plot
            df_res, _ = evaluate_var_predictions(
                model, prep, eval_df, feature_cols, q=0.05,
                label='Diagnostics OOS (20%)', return_details=True,
                save_plot_path=(str(oss_dir / 'diagnostics_series.png') if args.plots and args.diagnostics else None), do_plot=args.plots and args.diagnostics,
            )
            # Compare-Series (inkl. Baselines) – nach Baseline-Erzeugung zeichnen

            # Baselines für Diagnostics ergänzen (GARCH/LSTM)
            try:
                # GARCH auf Trainingsretouren fitten und als konstante 1-Step-VaR über das Eval-Fenster verwenden
                ret_series_tr = train_df['returns'].dropna() * 100.0
                if len(ret_series_tr) > 100:
                    garch_model = arch_model(ret_series_tr, vol='GARCH', p=1, q=1, dist='normal')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        garch_res = garch_model.fit(disp='off')
                    garch_forecast = garch_res.forecast(horizon=1)
                    cond_mean = float(garch_forecast.mean.iloc[-1, 0]) / 100.0
                    cond_vol = float(np.sqrt(garch_forecast.variance.iloc[-1, 0])) / 100.0
                    z_alpha = _norm_ppf(0.05)
                    df_res['GARCH_VaR'] = (cond_mean + z_alpha * cond_vol)
                else:
                    df_res['GARCH_VaR'] = np.nan
            except Exception:
                df_res['GARCH_VaR'] = np.nan

            try:
                # LSTM-Quantilmodell auf Trainingsretouren; Vorhersage über das gesamte Universum, dann auf Eval-Index reindexen
                seq = prep['seq_length']
                ret_train = train_df['returns'].dropna().values.astype(np.float32)
                if len(ret_train) > (seq + 100):
                    scaler = MinMaxScaler()
                    rs = ret_train.reshape(-1, 1)
                    rs_scaled = scaler.fit_transform(rs)
                    from tensorflow.keras.layers import LSTM as K_LSTM
                    lstm_model = tf.keras.Sequential([
                        K_LSTM(32, return_sequences=True, input_shape=(seq, 1)),
                        K_LSTM(16),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer='adam', loss=quantile_loss(0.05))
                    X_lstm, y_lstm = create_sequences(rs_scaled, seq)
                    lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32, verbose=0, shuffle=False)
                    ret_full = df_master['returns'].dropna().values.astype(np.float32)
                    rs_full = scaler.transform(ret_full.reshape(-1, 1))
                    X_full, _ = create_sequences(rs_full, seq)
                    preds_full = lstm_model.predict(X_full, verbose=0).flatten()
                    preds_full = scaler.inverse_transform(preds_full.reshape(-1, 1)).flatten()
                    idx_full = df_master['returns'].dropna().index[seq:]
                    lstm_ser = pd.Series(preds_full, index=idx_full).reindex(df_res.index)
                    df_res['LSTM_VaR'] = lstm_ser.values
                else:
                    df_res['LSTM_VaR'] = np.nan
            except Exception:
                df_res['LSTM_VaR'] = np.nan

            # Optional: Post-hoc Kalibrierung auch für Diagnostics
            if args.calibration in ('scale', 'conformal'):
                # Kalibrierungsfenster aus dem Ende der Trainingsperiode
                df_cal_all, _ = evaluate_var_predictions(model, prep, train_df, feature_cols, q=0.05, label='Diagnostics Calibration (train)', return_details=True)
                df_cal = df_cal_all.tail(args.calib_window)
                if not df_cal.empty and len(df_cal) >= max(30, args.seq_length + 5):
                    y_cal = df_cal['Return'].values
                    p_cal = df_cal['VaR'].values
                    y_eval = df_res['Return'].values
                    p_eval = df_res['VaR'].values

                    if args.calibration == 'scale':
                        grid = np.linspace(args.calib_scale_min, args.calib_scale_max, 61)
                        coverages = [np.mean(y_cal < s * p_cal) for s in grid]
                        s = float(grid[int(np.argmin(np.abs(np.array(coverages) - 0.05)))])
                        p_eval_cal = s * p_eval
                    else:  # conformal
                        shift = float(np.quantile(y_cal - p_cal, 0.05))
                        p_eval_cal = p_eval + shift

                    # Update Diagnostics-Resultate mit kalibriertem VaR
                    df_res['VaR'] = p_eval_cal
                    df_res['Breach'] = df_res['Return'].values < df_res['VaR'].values
                else:
                    print('ℹ️  Diagnostics: Zu wenige Punkte im Kalibrierungsfenster – keine Kalibrierung angewendet.')

            # Panels: einmal Transformer-only und einmal Compare
            if args.plots and args.diagnostics:
                plot_transformer_diagnostics(df_res, eval_df, q=0.05, vol_window=21, save_path=str(oss_dir / 'diagnostics_panels.png'), include_baselines=False)
                plot_transformer_diagnostics(df_res, eval_df, q=0.05, vol_window=21, save_path=str(oss_dir / 'diagnostics_panels_compare.png'), include_baselines=True)
            # Compare-Series (inkl. Baselines) final zeichnen
            try:
                fig_cs = plt.figure(figsize=(14, 6))
                q_val = 0.05
                idx = df_res.index
                plt.plot(idx, df_res['Return'].values, label='Daily Returns', color='tab:blue', linewidth=1.2)
                plt.plot(idx, df_res['VaR'].values, label=f'Transformer VaR {int((1-q_val)*100)}%', color='red', linewidth=1.4)
                if 'GARCH_VaR' in df_res.columns and not df_res['GARCH_VaR'].isna().all():
                    plt.plot(idx, df_res['GARCH_VaR'].values, label=f'GARCH VaR {int((1-q_val)*100)}%', color='green', linestyle='--', linewidth=1.2)
                if 'LSTM_VaR' in df_res.columns and not df_res['LSTM_VaR'].isna().all():
                    plt.plot(idx, df_res['LSTM_VaR'].values, label=f'LSTM VaR {int((1-q_val)*100)}%', color='orange', linestyle=':', linewidth=1.2)
                if 'HIST_VaR' in df_res.columns and not df_res['HIST_VaR'].isna().all():
                    plt.plot(idx, df_res['HIST_VaR'].values, label=f'Hist VaR {int((1-q_val)*100)}%', color='tab:purple', linestyle='-.', linewidth=1.2)
                plt.title('OSS (20%) – Daily Returns vs. VaR (Transformer/GARCH/LSTM)')
                plt.xlabel('Datum'); plt.ylabel('Rendite / VaR')
                plt.legend(); plt.grid(True); plt.tight_layout()
                if args.plots and args.diagnostics:
                    fig_cs.savefig(oss_dir / 'diagnostics_series_compare.png', dpi=150)
                plt.close(fig_cs)
            except Exception as e:
                print(f"⚠️  Konnte Diagnostics-Compare-Series nicht speichern ({e})")

            # Rolling Coverage (Transformer only) für OSS (20)
            try:
                if args.plots and args.diagnostics:
                    plot_rolling_coverage(df_res, q=0.05, window=63, save_path=oss_dir / 'diagnostics_rolling_coverage.png')
            except Exception as e:
                logging.getLogger("var_run").warning(f"Diagnostics Rolling Coverage Problem: {e}")
            # Rolling Coverage Compare (falls Baselines vorhanden)
            try:
                if args.plots and args.diagnostics:
                    plot_rolling_coverage_compare(df_res, q=0.05, window=63, save_path=oss_dir / 'diagnostics_rolling_coverage_compare.png')
            except Exception as e:
                logging.getLogger("var_run").warning(f"Diagnostics Rolling Coverage Compare Problem: {e}")
            # Makro-Plots entfernt
            df_res.to_csv(oss_dir / 'diagnostics_series.csv')
            # Z-Scores für Diagnostics (Train/OOS) speichern
            try:
                feats = [f for f in prep['features'] if f in df_master.columns]
                if prep.get('use_zscore', True) and len(feats) > 0:
                    z_tr = rolling_zscore(train_df[feats].ffill(), window=prep['z_window']).replace([np.inf, -np.inf], np.nan)
                    z_tr.to_csv(oss_dir / 'diagnostics_zscore_train.csv')
                    z_oos_full = rolling_zscore(df_master[feats].ffill(), window=prep['z_window']).replace([np.inf, -np.inf], np.nan)
                    z_oos = z_oos_full.reindex(df_res.index)
                    z_oos.to_csv(oss_dir / 'diagnostics_zscore_oos.csv')
            except Exception as e:
                logging.getLogger("var_run").warning(f"Diagnostics Z-Score Export Problem: {e}")

            df_clean = eval_df.copy()
            target = df_clean['returns'].dropna()
            feats_in_model = [f for f in prep['features'] if f in df_clean.columns]
            # Alle Features, die untersucht werden sollen (auch wenn nicht im Modell):
            all_feats = [f for f in feature_cols if f in df_clean.columns]
            all_inputs = df_clean[feats_in_model].loc[target.index]
            all_inputs = all_inputs.ffill()
            if prep.get('use_zscore', True):
                X_eval = rolling_zscore(all_inputs, window=prep['z_window'])
            else:
                X_eval = all_inputs.dropna(how='all')
            # Index konsistent schneiden
            target = target.loc[X_eval.index]
            seq_len = prep['seq_length']
            X_pred, _ = create_sequences(X_eval.values.astype(np.float32), seq_len)
            X_pred = X_pred.reshape((X_pred.shape[0], seq_len, len(feats_in_model)))
            _, y_true = create_sequences(target.values.reshape(-1, 1), seq_len)
            y_true = y_true.flatten()

            if len(X_pred) > 0:
                sample = min(1024, X_pred.shape[0])
                perm_df = permutation_importance_pinball(
                    model,
                    X_pred[:sample],
                    y_true[:sample],
                    feats_in_model,
                    q=0.05,
                    n_repeats=5,
                )
                # Füge nicht genutzte Features mit Degradation 0 hinzu, damit alle feature_cols abgedeckt sind
                other_feats = [f for f in all_feats if f not in feats_in_model]
                if len(other_feats) > 0:
                    zeros = pd.DataFrame({'feature': other_feats, 'pinball_degradation': [0.0] * len(other_feats)})
                    perm_df = pd.concat([perm_df, zeros], ignore_index=True)
                perm_df = perm_df.sort_values('pinball_degradation', ascending=False)
                # Write Diagnostics PI into OSS folder
                perm_df.to_csv(oss_dir / 'permutation_importance.csv', index=False)
                print("\n📈 Permutationsimportance (Top 10):\n", perm_df.head(10))
                # Visualisierung nur, wenn Plots aktiviert
                if args.plots and args.diagnostics:
                    try:
                        fig_pi = plt.figure(figsize=(10, 6))
                        top = perm_df.head(20).iloc[::-1]
                        plt.barh(top['feature'].astype(str), top['pinball_degradation'].astype(float), color='tab:gray')
                        plt.xlabel('Pinball-Degradation (Δ-Loss)'); plt.ylabel('Feature')
                        plt.title('Permutation Importance – Top 20 (n_repeats=5)')
                        # Save Diagnostics PI plot under OSS folder too to avoid confusion
                        plt.tight_layout(); fig_pi.savefig(oss_dir / 'permutation_importance.png', dpi=150)
                        plt.close(fig_pi)
                    except Exception as e:
                        print(f"⚠️  Konnte Permutation Importance Plot nicht speichern ({e})")
            else:
                print("⚠️  Überspringe Permutation Importance: zu wenige Sequenzen im OOS-Fenster.")

        # Metrikenplots (Balkendiagramme + Heatmap) erzeugen
        try:
            metrics_csv = run_dir / 'wf_metrics.csv'
            if metrics_csv.exists() and args.plots:
                mdf = pd.read_csv(metrics_csv)
                # Balkendiagramme pro Kennzahl
                fig, axes = plt.subplots(3, 3, figsize=(16, 12))
                metrics_cols = [
                    ('breach_rate', 'Breach-Rate'),
                    ('coverage_error', 'Coverage-Fehler'),
                    ('cvar', 'CVaR'),
                    ('rmse', 'RMSE'),
                    ('mae', 'MAE'),
                    ('pinball', 'Pinball-Loss'),
                    ('brier', 'Brier-Score'),
                    ('kupiec_p', 'Kupiec p-Wert'),
                    ('christoffersen_cc_p', 'Christoffersen CC p-Wert'),
                ]
                for ax, (col, title) in zip(axes.ravel(), metrics_cols):
                    ax.bar(mdf['period'].astype(str), mdf[col].astype(float), color='tab:gray')
                    ax.set_title(title)
                    ax.set_xlabel('Periode')
                    ax.set_ylabel(title)
                    ax.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / 'metrics_bars.png', dpi=150)
                plt.close(fig)

                # Heatmap über Kennzahlen x Perioden
                hm_cols = ['breach_rate','coverage_error','cvar','rmse','mae','pinball','brier','kupiec_p','christoffersen_cc_p']
                data = mdf.set_index('period')[hm_cols].astype(float)
                fig, ax = plt.subplots(figsize=(14, 6))
                im = ax.imshow(data.values, aspect='auto', cmap='viridis')
                ax.set_yticks(range(len(data.index)))
                ax.set_yticklabels(data.index.astype(str))
                ax.set_xticks(range(len(hm_cols)))
                ax.set_xticklabels(['Breach','CovErr','CVaR','RMSE','MAE','Pinball','Brier','Kupiec p','CC p'], rotation=45, ha='right')
                ax.set_title('Metriken-Heatmap (Perioden × Kennzahlen)')
                fig.colorbar(im, ax=ax, label='Skalierter Wert')
                plt.tight_layout()
                plt.savefig(plots_dir / 'metrics_heatmap.png', dpi=150)
                plt.close(fig)

            # Compare-Varianten, falls vorhanden
            metrics_cmp = run_dir / 'wf_metrics_compare.csv'
            if metrics_cmp.exists() and args.plots:
                cdf = pd.read_csv(metrics_cmp)
                models = cdf['model'].unique()
                hm_cols = ['breach_rate','coverage_error','cvar','rmse','mae','pinball','brier','kupiec_p','christoffersen_cc_p']
                # Balken: pro Kennzahl gruppiert (Periode × Modell)
                fig, axes = plt.subplots(3, 3, figsize=(18, 12))
                for ax, col in zip(axes.ravel(), hm_cols):
                    pivot = cdf.pivot(index='period', columns='model', values=col)
                    pivot.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_title(f"{col}")
                    ax.set_xlabel('Periode'); ax.set_ylabel(col)
                    ax.grid(True, axis='y', alpha=0.3)
                    ax.legend(title='Modell')
                plt.tight_layout()
                plt.savefig(plots_dir / 'metrics_bars_compare.png', dpi=150)
                plt.close(fig)

                # Heatmap: Zeilen=Periode×Modell, Spalten=Kennzahlen
                fig, ax = plt.subplots(figsize=(16, 8))
                cdf['row'] = cdf['period'].astype(str) + ' – ' + cdf['model']
                data = cdf.set_index('row')[hm_cols].astype(float)
                im = ax.imshow(data.values, aspect='auto', cmap='viridis')
                ax.set_yticks(range(len(data.index)))
                ax.set_yticklabels(data.index)
                ax.set_xticks(range(len(hm_cols)))
                ax.set_xticklabels(['Breach','CovErr','CVaR','RMSE','MAE','Pinball','Brier','Kupiec p','CC p'], rotation=45, ha='right')
                ax.set_title('Metriken-Heatmap (Periode × Modell × Kennzahl)')
                fig.colorbar(im, ax=ax, label='Skalierter Wert')
                plt.tight_layout()
                plt.savefig(plots_dir / 'metrics_heatmap_compare.png', dpi=150)
                plt.close(fig)
        except Exception as e:
            logging.getLogger("var_run").warning(f"Metriken-Visualisierung fehlgeschlagen: {e}")

    # ---------------------- Aufräumen: entferne nicht benötigte Aggregate ----------------------
    try:
        def _rm(p: Path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        # Bildartefakte nicht automatisch löschen; nur CSV-Spiegel entfernen
        # CSVs, die nur das letzte OOS-Jahr spiegeln, stets entfernen
        _rm(run_dir / 'wf_permutation_importance.csv')
        _rm(run_dir / 'wf_metrics_compare.csv')
    except Exception:
        pass