
"""
Quality-Aware and Uncertainty-Aware ETF Risk Monitoring as a Big Data Service
============================================================================

Extended version with fallback/component experiments and paper-aligned
uncertainty diagnostics.

What is added relative to the baseline walk-forward service
-----------------------------------------------------------
1. Keeps the main walk-forward service backtest intact.
2. Adds clean-input and corrupted-input fallback/component experiments.
3. Restores the focused corrupted-input quality-validation ablation used by
   the paper, exporting:
       - quality_validation_summary.csv
       - quality_validation_key_metrics.csv
4. Preserves the original fixed-threshold uncertainty buckets only for
   internal alert escalation, now exported as:
       - uncertainty_flag_alert_legacy
5. Adds the paper-facing diagnostic uncertainty state:
       - uncertainty_state = rolling low / elevated
   computed symbol-by-symbol from the rolling 90th percentile of prior
   uncertainty scores, using up to the most recent 252 evaluable dates.
6. Clears legacy uncertainty summaries from evaluation outputs by reporting:
       - uncertainty_low_safe_var
       - uncertainty_elevated_safe_var
       - uncertainty_low / uncertainty_elevated activation counts

Designed for:
- C:\\Users\\26876\\Desktop\\2026BIgdataservice
- Files named:
    multiasset_daily_10y_panel_model.csv
    VIXCLS.csv
    zero_coupon_yield.csv
- Optional:
    VXVCLS.csv

Author:
OpenAI for Hanson
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    base_dir: Path = Path(
        os.environ.get("BIGDATASERVICE_BASE_DIR", r"C:\Users\26876\Desktop\2026BIgdataservice")
    )
    output_dir_name: str = "risk_monitoring_service_outputs_revised8_fallback_ladder_qualityval"

    # Risk target
    alpha: float = 0.05
    target_col: str = "ret_next"

    # Walk-forward setup
    train_window_days: int = 756
    calib_window_days: int = 63
    retrain_every: int = 63
    test_start: str = "2023-01-03"

    # Model / uncertainty
    n_ensemble: int = 5
    bootstrap_frac_dates: float = 0.80
    pca_max_components: int = 8
    pca_explained_variance: float = 0.95
    ood_quantile_ref: float = 0.95

    # Paper-facing uncertainty diagnostics
    uncertainty_diag_window: int = 252
    uncertainty_diag_quantile: float = 0.90

    # Baseline / fallback
    hist_var_window: int = 252
    safe_hist_var_window: int = 63
    recent_breach_window: int = 60
    fallback_lambda_u: float = 0.75
    fallback_lambda_q: float = 0.50

    # Alerting
    orange_fallback_ratio: float = 0.35
    red_fallback_ratio: float = 0.75
    orange_drift_score: float = 0.50
    red_drift_score: float = 1.00

    # Quality weights
    q_missing: float = 0.30
    q_invalid_ohlc: float = 0.35
    q_extreme_jump: float = 0.15
    q_volume_anomaly: float = 0.10
    q_stale: float = 0.10

    # Model hyperparameters
    model_learning_rate: float = 0.05
    model_max_iter: int = 150
    model_max_depth: int = 5
    model_min_samples_leaf: int = 20
    model_l2_regularization: float = 1e-3

    # Fallback/component experiments
    run_component_ablation_experiment: bool = True
    run_corrupted_component_experiment: bool = True
    run_quality_validation_experiment: bool = True
    corruption_rate: float = 0.15
    corruption_seed: int = 20260405
    corruption_modes: Tuple[str, ...] = ("missing", "stale", "ohlc")
    save_component_experiment_daily_outputs: bool = False
    save_quality_validation_daily_outputs: bool = True

    # Experiment hygiene
    drift_reference_design: str = "full_service"
    recompute_cross_asset_after_corruption: bool = True


# ============================================================
# Small helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_input_file(base_dir: Path, stem: str) -> Path:
    candidates = [
        base_dir / f"{stem}.csv",
        base_dir / f"{stem}.xlsx",
        base_dir / f"{stem}.xls",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find file for stem={stem!r} in {base_dir}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def sigmoid_rescale(x: pd.Series, center: float, scale: float) -> pd.Series:
    z = (x - center) / max(scale, 1e-8)
    return 1.0 / (1.0 + np.exp(-z))


def kupiec_test(exceedances: np.ndarray, alpha: float) -> Dict[str, float]:
    exceedances = np.asarray(exceedances).astype(float)
    exceedances = exceedances[~np.isnan(exceedances)]
    n = int(exceedances.size)
    x = int(exceedances.sum())
    if n == 0:
        return {"n": 0, "x": 0, "empirical_rate": np.nan, "lr_uc": np.nan}
    p_hat = x / n
    if p_hat in {0.0, 1.0}:
        lr = np.nan
    else:
        ll0 = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
        ll1 = (n - x) * math.log(1 - p_hat) + x * math.log(p_hat)
        lr = -2 * (ll0 - ll1)
    return {"n": n, "x": x, "empirical_rate": p_hat, "lr_uc": lr}


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true)
    q_pred = np.asarray(q_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(q_pred))
    if not mask.any():
        return np.nan
    delta = y_true[mask] - q_pred[mask]
    loss = np.where(delta >= 0, alpha * delta, (alpha - 1.0) * delta)
    return float(np.nanmean(loss))


def safe_median(series: pd.Series, default: float = 0.0) -> float:
    val = series.median(skipna=True)
    if pd.isna(val):
        return float(default)
    return float(val)


def get_feature_medians(df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    medians = {}
    for c in feature_cols:
        medians[c] = safe_median(pd.to_numeric(df[c], errors="coerce"), default=0.0)
    return pd.Series(medians)


def prepare_X(df: pd.DataFrame, feature_cols: List[str], fill_values: pd.Series) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(fill_values)
    return X


def quantile_offset_from_calibration(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    mask = (~np.isnan(y_true)) & (~np.isnan(q_pred))
    if mask.sum() == 0:
        return 0.0
    resid = y_true[mask] - q_pred[mask]
    return float(np.nanquantile(resid, alpha))


def bootstrap_by_date(train_df: pd.DataFrame, rng: np.random.RandomState, frac_dates: float) -> pd.DataFrame:
    unique_dates = np.array(sorted(train_df["date"].dropna().unique()))
    if unique_dates.size == 0:
        return train_df.copy()

    n_pick = max(1, int(np.ceil(frac_dates * unique_dates.size)))
    sampled_dates = rng.choice(unique_dates, size=n_pick, replace=True)
    sampled_dates = pd.to_datetime(sampled_dates)

    counts = pd.Series(sampled_dates).value_counts()
    tmp = train_df.merge(
        counts.rename("rep_count"),
        left_on="date",
        right_index=True,
        how="inner",
    )
    if tmp.empty:
        return train_df.sample(
            frac=1.0, replace=True, random_state=int(rng.randint(1_000_000))
        ).reset_index(drop=True)

    boot = tmp.loc[tmp.index.repeat(tmp["rep_count"].astype(int))].drop(columns=["rep_count"])
    boot = boot.sample(frac=1.0, random_state=int(rng.randint(1_000_000))).reset_index(drop=True)
    return boot


# ============================================================
# Data loading
# ============================================================

def load_main_panel(cfg: Config) -> pd.DataFrame:
    path = find_input_file(cfg.base_dir, "multiasset_daily_10y_panel_model")
    df = read_table(path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    numeric_cols = [
        "open", "high", "low", "close", "volume", "ret", "ret_next",
        "rolling_vol_5", "rolling_vol_10", "rolling_vol_20", "ewma_vol_20",
        "parkinson_proxy", "gk_proxy", "cum_max_close", "drawdown"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby("symbol", group_keys=False)
    df["prev_close"] = g["close"].shift(1)
    df["gap"] = df["open"] / df["prev_close"] - 1.0
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_move"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["volume_log"] = np.log1p(df["volume"])

    for w in [3, 5, 10, 20, 60]:
        df[f"ret_mean_{w}"] = g["ret"].transform(lambda s: s.rolling(w).mean())
        df[f"ret_std_{w}"] = g["ret"].transform(lambda s: s.rolling(w).std())
        df[f"close_ma_ratio_{w}"] = df["close"] / g["close"].transform(lambda s: s.rolling(w).mean())
        df[f"volume_z_{w}"] = (
            (df["volume_log"] - g["volume_log"].transform(lambda s: s.rolling(w).mean()))
            / g["volume_log"].transform(lambda s: s.rolling(w).std())
        )

    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag_{lag}"] = g["ret"].shift(lag)
        df[f"vol20_lag_{lag}"] = g["rolling_vol_20"].shift(lag)

    stale_count = []
    for _, sub in df.groupby("symbol"):
        cnt = 0
        vals = []
        prev = None
        for close in sub["close"].to_numpy():
            if prev is not None and abs(close - prev) < 1e-12:
                cnt += 1
            else:
                cnt = 0
            vals.append(cnt)
            prev = close
        stale_count.extend(vals)
    df["stale_count"] = stale_count

    return df


def load_vix(cfg: Config) -> pd.DataFrame:
    path = find_input_file(cfg.base_dir, "VIXCLS")
    df = read_table(path).copy()
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    val_col = [c for c in df.columns if c.upper().startswith("VIX")][0]
    df = df.rename(columns={date_col: "date", val_col: "vix"})
    df["date"] = pd.to_datetime(df["date"])
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.sort_values("date")
    df["vix_change_1"] = df["vix"].pct_change()
    df["vix_ma20_ratio"] = df["vix"] / df["vix"].rolling(20).mean()
    df["vix_z20"] = (df["vix"] - df["vix"].rolling(20).mean()) / df["vix"].rolling(20).std()
    return df[["date", "vix", "vix_change_1", "vix_ma20_ratio", "vix_z20"]]


def load_optional_vxv(cfg: Config) -> Optional[pd.DataFrame]:
    try:
        path = find_input_file(cfg.base_dir, "VXVCLS")
    except FileNotFoundError:
        return None
    df = read_table(path).copy()
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    value_cols = [c for c in df.columns if c != date_col]
    val_col = value_cols[0]
    df = df.rename(columns={date_col: "date", val_col: "vxv"})
    df["date"] = pd.to_datetime(df["date"])
    df["vxv"] = pd.to_numeric(df["vxv"], errors="coerce")
    return df[["date", "vxv"]].sort_values("date")


def nearest_curve_rates(curve: pd.DataFrame, targets: List[int]) -> pd.DataFrame:
    rows = []
    for d, sub in curve.groupby("date"):
        days = sub["days"].to_numpy()
        rates = sub["rate"].to_numpy()
        out = {"date": d}
        for t in targets:
            idx = int(np.argmin(np.abs(days - t)))
            out[f"zc_{t}d"] = float(rates[idx])
        rows.append(out)
    wide = pd.DataFrame(rows).sort_values("date")
    if {"zc_30d", "zc_365d"}.issubset(wide.columns):
        wide["term_spread_1y_1m"] = wide["zc_365d"] - wide["zc_30d"]
    if {"zc_365d", "zc_1825d"}.issubset(wide.columns):
        wide["term_spread_5y_1y"] = wide["zc_1825d"] - wide["zc_365d"]
    return wide


def load_curve(cfg: Config) -> pd.DataFrame:
    path = find_input_file(cfg.base_dir, "zero_coupon_yield")
    df = read_table(path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["days"] = pd.to_numeric(df["days"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["date", "days", "rate"])
    targets = [30, 90, 365, 1825]
    return nearest_curve_rates(df, targets)


def merge_macro_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    vix = load_vix(cfg)
    curve = load_curve(cfg)

    out = panel.merge(vix, on="date", how="left")
    out = out.merge(curve, on="date", how="left")

    out["vix_available"] = out["vix"].notna().astype(int)
    out["curve_available"] = out["zc_30d"].notna().astype(int)

    for c in ["zc_30d", "zc_90d", "zc_365d", "zc_1825d", "term_spread_1y_1m", "term_spread_5y_1y"]:
        if c in out.columns:
            out[f"{c}_missing"] = out[c].isna().astype(int)

    vxv = load_optional_vxv(cfg)
    if vxv is not None:
        out = out.merge(vxv, on="date", how="left")
        out["vxv_available"] = out["vxv"].notna().astype(int)
        out["vix_vxv_ratio"] = out["vix"] / out["vxv"]
    else:
        out["vxv_available"] = 0

    return out


# ============================================================
# Quality layer
# ============================================================

def add_quality_layer(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    critical = ["open", "high", "low", "close", "volume", "ret"]
    critical = [c for c in critical if c in out.columns]
    out["q_missing_score"] = out[critical].isna().mean(axis=1)

    invalid_ohlc = (
        (out["low"] > out[["open", "close"]].min(axis=1)) |
        (out["high"] < out[["open", "close"]].max(axis=1)) |
        (out["high"] < out["low"])
    )
    out["q_invalid_ohlc_score"] = invalid_ohlc.astype(float)

    ret_abs = out["ret"].abs()
    ret_z = out.groupby("symbol")["ret"].transform(
        lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std()
    )
    out["ret_z_60"] = ret_z
    jump_score = np.maximum(
        (ret_abs > 0.15).astype(float),
        sigmoid_rescale(ret_z.abs().fillna(0.0), center=3.0, scale=1.0)
    )
    out["q_extreme_jump_score"] = jump_score.clip(0.0, 1.0)

    out["q_volume_anomaly_score"] = sigmoid_rescale(
        out["volume_z_20"].abs().fillna(0.0), center=3.0, scale=1.0
    ).clip(0.0, 1.0)

    out["q_stale_score"] = (out["stale_count"] >= 2).astype(float)

    score = (
        cfg.q_missing * out["q_missing_score"].fillna(0.0)
        + cfg.q_invalid_ohlc * out["q_invalid_ohlc_score"].fillna(0.0)
        + cfg.q_extreme_jump * out["q_extreme_jump_score"].fillna(0.0)
        + cfg.q_volume_anomaly * out["q_volume_anomaly_score"].fillna(0.0)
        + cfg.q_stale * out["q_stale_score"].fillna(0.0)
    )
    out["quality_score"] = score.clip(0.0, 1.0)

    out["quality_flag"] = pd.cut(
        out["quality_score"],
        bins=[-np.inf, 0.25, 0.60, np.inf],
        labels=["green", "yellow", "red"],
    ).astype(str)

    return out


def rebuild_service_observable_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute same-day observable features after synthetic corruption.
    Only touches prediction-time observable quantities.
    """
    out = df.copy()

    if {"open", "prev_close"}.issubset(out.columns):
        out["gap"] = out["open"] / out["prev_close"].replace(0, np.nan) - 1.0

    if {"high", "low", "close"}.issubset(out.columns):
        out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)

    if {"close", "open"}.issubset(out.columns):
        out["oc_move"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    if "volume" in out.columns:
        vol = pd.to_numeric(out["volume"], errors="coerce")
        vol = vol.where(vol >= 0, np.nan)
        out["volume"] = vol
        out["volume_log"] = np.log1p(vol)

    return out


def recompute_current_day_cross_asset_features(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild date-level cross-asset aggregates after corruption is applied."""
    out = pred_df.copy()
    daily = (
        out.groupby("date")
        .agg(
            mkt_ret=("ret", "mean"),
            mkt_vol20=("rolling_vol_20", "mean"),
            cross_dispersion=("ret", "std"),
            cross_drawdown=("drawdown", "mean"),
            cross_hl_range=("hl_range", "mean"),
        )
        .reset_index()
    )
    for c in ["mkt_ret", "mkt_vol20", "cross_dispersion", "cross_drawdown", "cross_hl_range"]:
        if c in out.columns:
            out = out.drop(columns=[c])
    out = out.merge(daily, on="date", how="left")
    return out


def apply_prediction_time_corruption(pred_df: pd.DataFrame, cfg: Config, current_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Apply service-time corruption only to the prediction rows for the current day.
    This makes the experiment realistic: the model is still trained on historical data,
    but the live input stream can be partially degraded.
    """
    out = pred_df.copy()
    out["corruption_applied"] = 0
    out["corruption_mode"] = "none"

    if len(out) == 0:
        return out

    seed = cfg.corruption_seed + int(pd.Timestamp(current_ts).strftime("%Y%m%d"))
    rng = np.random.RandomState(seed)

    mask = rng.rand(len(out)) < cfg.corruption_rate
    if mask.sum() == 0 and len(out) > 0 and rng.rand() < 0.50:
        mask[rng.randint(len(out))] = True

    idx_positions = np.where(mask)[0]
    if len(idx_positions) == 0:
        return out

    raw_obs_cols = [c for c in ["open", "high", "low", "close", "volume", "ret"] if c in out.columns]
    volume_z_cols = [c for c in out.columns if c.startswith("volume_z_")]

    for pos in idx_positions:
        row_idx = out.index[pos]
        mode = rng.choice(cfg.corruption_modes)
        out.at[row_idx, "corruption_applied"] = 1
        out.at[row_idx, "corruption_mode"] = mode

        if mode == "missing":
            n_drop = max(2, min(4, len(raw_obs_cols)))
            drop_cols = list(rng.choice(raw_obs_cols, size=n_drop, replace=False))
            for c in drop_cols:
                out.at[row_idx, c] = np.nan

            for c in ["gap", "hl_range", "oc_move", "volume_log"] + volume_z_cols:
                if c in out.columns:
                    out.at[row_idx, c] = np.nan

        elif mode == "stale":
            prev_close = out.at[row_idx, "prev_close"] if "prev_close" in out.columns else np.nan
            ref = prev_close
            if pd.isna(ref):
                ref = out.at[row_idx, "close"] if "close" in out.columns else np.nan

            for c in ["open", "high", "low", "close"]:
                if c in out.columns:
                    out.at[row_idx, c] = ref
            if "ret" in out.columns:
                out.at[row_idx, "ret"] = 0.0
            if "gap" in out.columns:
                out.at[row_idx, "gap"] = 0.0
            if "hl_range" in out.columns:
                out.at[row_idx, "hl_range"] = 0.0
            if "oc_move" in out.columns:
                out.at[row_idx, "oc_move"] = 0.0
            if "stale_count" in out.columns:
                out.at[row_idx, "stale_count"] = max(2, float(out.at[row_idx, "stale_count"]))

        elif mode == "ohlc":
            open_px = out.at[row_idx, "open"] if "open" in out.columns else np.nan
            close_px = out.at[row_idx, "close"] if "close" in out.columns else np.nan
            if pd.isna(open_px):
                open_px = 100.0
                if "open" in out.columns:
                    out.at[row_idx, "open"] = open_px
            if pd.isna(close_px):
                close_px = open_px

            low_bad = max(open_px, close_px) + max(0.01 * abs(close_px), 1e-3)
            high_bad = min(open_px, close_px) - max(0.01 * abs(close_px), 1e-3)

            if "low" in out.columns:
                out.at[row_idx, "low"] = low_bad
            if "high" in out.columns:
                out.at[row_idx, "high"] = high_bad
            if "hl_range" in out.columns:
                out.at[row_idx, "hl_range"] = (high_bad - low_bad) / max(abs(close_px), 1e-8)

    out = rebuild_service_observable_features(out)
    if cfg.recompute_cross_asset_after_corruption:
        out = recompute_current_day_cross_asset_features(out)
    out = add_quality_layer(out, cfg)
    return out


# ============================================================
# Cross-asset / baseline features
# ============================================================

def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    daily = (
        out.groupby("date")
        .agg(
            mkt_ret=("ret", "mean"),
            mkt_vol20=("rolling_vol_20", "mean"),
            cross_dispersion=("ret", "std"),
            cross_drawdown=("drawdown", "mean"),
            cross_hl_range=("hl_range", "mean"),
        )
        .reset_index()
    )
    out = out.merge(daily, on="date", how="left")
    out["symbol_code"] = out["symbol"].astype("category").cat.codes
    return out


def add_baseline_vars(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol", group_keys=False)

    out["hist_var_alpha"] = g["ret"].transform(
        lambda s: s.shift(1).rolling(cfg.hist_var_window).quantile(cfg.alpha)
    )
    z = -1.6448536269514729
    out["ewma_norm_var"] = -z * out["ewma_vol_20"]
    out["ewma_norm_var"] = -out["ewma_norm_var"].abs()
    out["safe_hist_var"] = g["ret"].transform(
        lambda s: s.shift(1).rolling(cfg.safe_hist_var_window).quantile(cfg.alpha)
    )
    out["realized_vol_20"] = g["ret"].transform(lambda s: s.rolling(20).std())
    return out


def build_feature_columns(df: pd.DataFrame, use_quality_feature: bool = True) -> List[str]:
    cols = [
        "symbol_code",
        "ret",
        "rolling_vol_5", "rolling_vol_10", "rolling_vol_20", "ewma_vol_20",
        "parkinson_proxy", "gk_proxy", "drawdown",
        "gap", "hl_range", "oc_move",
        "ret_mean_3", "ret_mean_5", "ret_mean_10", "ret_mean_20", "ret_mean_60",
        "ret_std_3", "ret_std_5", "ret_std_10", "ret_std_20", "ret_std_60",
        "close_ma_ratio_3", "close_ma_ratio_5", "close_ma_ratio_10", "close_ma_ratio_20", "close_ma_ratio_60",
        "volume_log", "volume_z_3", "volume_z_5", "volume_z_10", "volume_z_20", "volume_z_60",
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_10",
        "vol20_lag_1", "vol20_lag_2", "vol20_lag_3", "vol20_lag_5", "vol20_lag_10",
        "vix", "vix_change_1", "vix_ma20_ratio", "vix_z20", "vix_available",
        "zc_30d", "zc_90d", "zc_365d", "zc_1825d",
        "term_spread_1y_1m", "term_spread_5y_1y", "curve_available",
        "zc_30d_missing", "zc_90d_missing", "zc_365d_missing", "zc_1825d_missing",
        "term_spread_1y_1m_missing", "term_spread_5y_1y_missing",
        "mkt_ret", "mkt_vol20", "cross_dispersion", "cross_drawdown", "cross_hl_range",
    ]
    if use_quality_feature:
        cols.append("quality_score")

    optional = ["vxv", "vix_vxv_ratio", "vxv_available"]
    cols.extend([c for c in optional if c in df.columns])
    return [c for c in cols if c in df.columns]


def build_ood_feature_columns(feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if c != "symbol_code"]


# ============================================================
# OOD / uncertainty helpers
# ============================================================

def mahalanobis_batch(X: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = X - mu
    md2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    md2 = np.maximum(md2, 0.0)
    return np.sqrt(md2)


def fit_ood_detector(X_train: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    n_features = Xs.shape[1]
    max_comp = min(cfg.pca_max_components, n_features, max(1, Xs.shape[0] - 1))
    pca_full = PCA(n_components=min(max_comp, n_features), random_state=0)
    Xp_full = pca_full.fit_transform(Xs)

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum, cfg.pca_explained_variance) + 1)
    k = max(1, min(k, Xp_full.shape[1]))

    pca = PCA(n_components=k, random_state=0)
    Xp = pca.fit_transform(Xs)
    mu = Xp.mean(axis=0)
    cov = np.cov(Xp, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.pinv(cov)

    d_train = mahalanobis_batch(Xp, mu, cov_inv)
    ref = float(np.nanquantile(d_train, cfg.ood_quantile_ref))
    return {
        "scaler": scaler,
        "pca": pca,
        "mu": mu,
        "cov_inv": cov_inv,
        "ref": max(ref, 1e-8),
    }


def score_ood(detector: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    Xs = detector["scaler"].transform(X)
    Xp = detector["pca"].transform(Xs)
    d = mahalanobis_batch(Xp, detector["mu"], detector["cov_inv"])
    score = np.clip(d / detector["ref"], 0.0, None)
    return np.clip((score - 1.0) / 1.5, 0.0, 1.0)


def recent_drift_score(
    results_history: pd.DataFrame,
    symbol: str,
    current_date: pd.Timestamp,
    cfg: Config,
    breach_col: str = "breach_safe",
) -> float:
    if results_history.empty or breach_col not in results_history.columns:
        return 0.0

    past = results_history[
        (results_history["symbol"] == symbol) &
        (results_history["date"] < current_date) &
        (results_history["is_evaluable"] == 1)
    ].tail(cfg.recent_breach_window)

    if len(past) < max(20, cfg.recent_breach_window // 2):
        return 0.0

    rate = float(past[breach_col].mean())
    excess = max(rate - cfg.alpha, 0.0)
    return float(min(1.0, excess / max(2.0 * cfg.alpha, 1e-8)))


def uncertainty_alert_flag_from_score(u: pd.Series) -> pd.Series:
    """
    Legacy fixed-threshold uncertainty buckets kept only for alert logic.
    """
    return pd.cut(
        u,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)


def add_uncertainty_diagnostic_state(
    pred_df: pd.DataFrame,
    history_results: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Paper-facing diagnostic uncertainty state:
    - computed symbol-by-symbol
    - uses strictly prior evaluable dates only
    - uses up to the most recent cfg.uncertainty_diag_window dates
    - elevated iff current uncertainty_score exceeds the rolling quantile cutoff
    """
    out = pred_df.copy()
    cutoffs: List[float] = []
    states: List[str] = []
    hist_ns: List[int] = []

    history_ok = (
        history_results is not None
        and not history_results.empty
        and {"symbol", "date", "is_evaluable", "uncertainty_score"}.issubset(history_results.columns)
    )

    for _, row in out.iterrows():
        cutoff = np.nan
        state = "low"
        hist_n = 0

        if history_ok:
            hist = history_results[
                (history_results["symbol"] == row["symbol"]) &
                (history_results["date"] < row["date"]) &
                (history_results["is_evaluable"] == 1)
            ]["uncertainty_score"].dropna().tail(cfg.uncertainty_diag_window)

            hist_n = int(len(hist))
            if hist_n > 0:
                cutoff = float(np.nanquantile(hist.to_numpy(), cfg.uncertainty_diag_quantile))
                state = "elevated" if float(row["uncertainty_score"]) > cutoff else "low"

        cutoffs.append(cutoff)
        states.append(state)
        hist_ns.append(hist_n)

    out["uncertainty_cutoff_u"] = cutoffs
    out["uncertainty_history_n"] = hist_ns
    out["uncertainty_state"] = states
    return out


# ============================================================
# Model
# ============================================================

def make_quantile_model(cfg: Config, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=cfg.alpha,
        learning_rate=cfg.model_learning_rate,
        max_iter=cfg.model_max_iter,
        max_depth=cfg.model_max_depth,
        min_samples_leaf=cfg.model_min_samples_leaf,
        l2_regularization=cfg.model_l2_regularization,
        random_state=seed,
        early_stopping=False,
    )


# ============================================================
# Main dataset builder
# ============================================================

def build_dataset(cfg: Config) -> pd.DataFrame:
    panel = load_main_panel(cfg)
    panel = merge_macro_features(panel, cfg)
    panel = add_quality_layer(panel, cfg)
    panel = add_cross_asset_features(panel)
    panel = add_baseline_vars(panel, cfg)

    essential_cols = [c for c in ["date", "symbol", "hist_var_alpha", "safe_hist_var"] if c in panel.columns]
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel = panel.dropna(subset=essential_cols).copy()

    return panel


# ============================================================
# Alerting logic
# ============================================================

def alert_from_row(row: pd.Series, cfg: Config, use_quality_alert: bool = True) -> str:
    alert_flag = row.get("uncertainty_flag_alert_legacy", "low")

    quality_red = use_quality_alert and (row["quality_flag"] == "red")
    quality_yellow = use_quality_alert and (row["quality_flag"] == "yellow")

    if (
        quality_red or
        alert_flag == "high" or
        row["u_drift"] >= cfg.red_drift_score or
        row["fallback_ratio"] >= cfg.red_fallback_ratio
    ):
        return "red"

    if (
        quality_yellow or
        alert_flag == "medium" or
        row["u_drift"] >= cfg.orange_drift_score or
        row["fallback_ratio"] >= cfg.orange_fallback_ratio
    ):
        return "orange"

    return "green"


def compute_fallback_design_predictions(pred_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = pred_df.copy()

    hist_anchor = out["safe_hist_var"].fillna(out["hist_var_alpha"])
    adj_scale = out["rolling_vol_20"].fillna(out["realized_vol_20"]).fillna(0.01)

    u_term = out["uncertainty_score"].fillna(0.0)
    q_term = out["quality_score"].fillna(0.0)

    simple_hist = np.minimum(out["var_pred"].to_numpy(), hist_anchor.to_numpy())

    uncertainty_adj = adj_scale * (cfg.fallback_lambda_u * u_term)
    quality_adj = adj_scale * (cfg.fallback_lambda_q * q_term)
    full_adj = adj_scale * (cfg.fallback_lambda_u * u_term + cfg.fallback_lambda_q * q_term)

    uncertainty_only = np.minimum.reduce([
        out["var_pred"].to_numpy(),
        hist_anchor.to_numpy(),
        (out["var_pred"] - uncertainty_adj).to_numpy(),
    ])
    quality_only = np.minimum.reduce([
        out["var_pred"].to_numpy(),
        hist_anchor.to_numpy(),
        (out["var_pred"] - quality_adj).to_numpy(),
    ])
    full_service = np.minimum.reduce([
        out["var_pred"].to_numpy(),
        hist_anchor.to_numpy(),
        (out["var_pred"] - full_adj).to_numpy(),
    ])

    out["safe_hist_anchor"] = hist_anchor
    out["adj_scale"] = adj_scale
    out["simple_fallback_pred"] = simple_hist
    out["uncertainty_only_pred"] = uncertainty_only
    out["quality_only_pred"] = quality_only
    out["full_fallback_pred"] = full_service
    out["raw_model_pred"] = out["var_pred"]
    out["uncertainty_adj"] = uncertainty_adj
    out["quality_adj"] = quality_adj
    out["full_adj"] = full_adj
    return out


def choose_service_output_column(fallback_design: str) -> str:
    mapping = {
        "raw_model": "raw_model_pred",
        "simple_fallback_only": "simple_fallback_pred",
        "uncertainty_only_fallback": "uncertainty_only_pred",
        "quality_only_fallback": "quality_only_pred",
        "full_service": "full_fallback_pred",
    }
    if fallback_design not in mapping:
        raise ValueError(f"Unknown fallback_design={fallback_design!r}. Expected one of {sorted(mapping)}")
    return mapping[fallback_design]


# ============================================================
# Walk-forward backtest
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    cfg: Config,
    experiment_name: str = "main_full_service",
    fallback_design: str = "full_service",
    use_quality_feature: bool = True,
    use_quality_alert: bool = True,
    corrupt_prediction_inputs: bool = False,
    drift_reference_history: Optional[pd.DataFrame] = None,
    drift_reference_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    dates = pd.Index(pd.to_datetime(df["date"]).dropna().unique()).sort_values()
    date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    test_start_ts = pd.Timestamp(cfg.test_start).normalize()
    test_dates = [pd.Timestamp(d).normalize() for d in dates if pd.Timestamp(d).normalize() >= test_start_ts]

    feature_cols = build_feature_columns(df, use_quality_feature=use_quality_feature)
    ood_feature_cols = build_ood_feature_columns(feature_cols)
    safe_col = choose_service_output_column(fallback_design)

    rows_out: List[pd.DataFrame] = []
    retrain_state = None
    results_so_far = pd.DataFrame()
    fixed_drift_reference = drift_reference_history is not None
    drift_reference_name = drift_reference_name or ("external_reference" if fixed_drift_reference else experiment_name)

    for current_ts in tqdm(test_dates, desc=f"Backtest[{experiment_name}]", ncols=100):
        current_idx = date_to_idx[current_ts]

        if current_idx < cfg.train_window_days + cfg.calib_window_days:
            continue

        need_retrain = (
            retrain_state is None or
            (current_idx - retrain_state["anchor_idx"] >= cfg.retrain_every)
        )

        if need_retrain:
            train_start_idx = current_idx - cfg.calib_window_days - cfg.train_window_days
            train_end_idx = current_idx - cfg.calib_window_days - 1
            calib_start_idx = current_idx - cfg.calib_window_days
            calib_end_idx = current_idx - 1

            train_dates = pd.to_datetime(dates[train_start_idx:train_end_idx + 1])
            calib_dates = pd.to_datetime(dates[calib_start_idx:calib_end_idx + 1])

            train_df = df[df["date"].isin(train_dates)].copy()
            train_df = train_df.dropna(subset=[cfg.target_col]).copy()

            calib_df = df[df["date"].isin(calib_dates)].copy()
            calib_df = calib_df.dropna(subset=[cfg.target_col]).copy()

            if len(train_df) < 500 or len(calib_df) < 100:
                continue

            fill_values = get_feature_medians(train_df, feature_cols)

            models = []
            calib_pred_cols = []

            for k in range(cfg.n_ensemble):
                rng = np.random.RandomState(123 + k)
                boot_df = bootstrap_by_date(train_df, rng, cfg.bootstrap_frac_dates)
                X_boot = prepare_X(boot_df, feature_cols, fill_values)
                y_boot = boot_df[cfg.target_col].to_numpy()

                model = make_quantile_model(cfg, seed=123 + k)
                model.fit(X_boot, y_boot)
                models.append(model)

                calib_pred_cols.append(model.predict(prepare_X(calib_df, feature_cols, fill_values)))

            calib_pred_mean = np.column_stack(calib_pred_cols).mean(axis=1)
            calib_offset = quantile_offset_from_calibration(
                calib_df[cfg.target_col].to_numpy(),
                calib_pred_mean,
                cfg.alpha,
            )

            detector = fit_ood_detector(
                prepare_X(train_df, ood_feature_cols, fill_values[ood_feature_cols]),
                cfg,
            )

            retrain_state = {
                "models": models,
                "detector": detector,
                "anchor_idx": current_idx,
                "feature_cols": feature_cols,
                "ood_feature_cols": ood_feature_cols,
                "fill_values": fill_values,
                "calib_offset": calib_offset,
                "train_start": str(train_dates.min().date()),
                "train_end": str(train_dates.max().date()),
                "calib_start": str(calib_dates.min().date()),
                "calib_end": str(calib_dates.max().date()),
            }

        pred_df = df[df["date"] == current_ts].copy()
        if pred_df.empty:
            continue

        pred_df["corruption_applied"] = 0
        pred_df["corruption_mode"] = "none"

        if corrupt_prediction_inputs:
            pred_df = apply_prediction_time_corruption(pred_df, cfg, current_ts)

        X_pred = prepare_X(pred_df, retrain_state["feature_cols"], retrain_state["fill_values"])
        X_pred_ood = prepare_X(
            pred_df,
            retrain_state["ood_feature_cols"],
            retrain_state["fill_values"][retrain_state["ood_feature_cols"]],
        )

        preds = np.column_stack([m.predict(X_pred) for m in retrain_state["models"]])
        pred_df["var_pred_raw"] = preds.mean(axis=1)
        pred_df["calib_offset"] = retrain_state["calib_offset"]
        pred_df["var_pred"] = pred_df["var_pred_raw"] + pred_df["calib_offset"]

        pred_df["u_model"] = preds.std(axis=1)
        pred_df["u_ood"] = score_ood(retrain_state["detector"], X_pred_ood)

        drift_history = drift_reference_history if fixed_drift_reference else results_so_far
        drift_scores = []
        for _, row in pred_df.iterrows():
            drift_scores.append(
                recent_drift_score(
                    drift_history,
                    row["symbol"],
                    current_ts,
                    cfg,
                    breach_col="breach_safe",
                )
            )
        pred_df["u_drift"] = drift_scores

        vol_ref = pred_df["rolling_vol_20"].replace(0, np.nan).fillna(pred_df["realized_vol_20"]).fillna(1e-4)
        pred_df["u_model_scaled"] = (pred_df["u_model"] / vol_ref).clip(0.0, 3.0) / 3.0

        pred_df["uncertainty_score"] = (
            0.40 * pred_df["u_model_scaled"].fillna(0.0)
            + 0.35 * pred_df["u_ood"].fillna(0.0)
            + 0.25 * pred_df["u_drift"].fillna(0.0)
        ).clip(0.0, 1.0)

        # Legacy fixed-threshold buckets kept only for alert escalation.
        pred_df["uncertainty_flag_alert_legacy"] = uncertainty_alert_flag_from_score(pred_df["uncertainty_score"])

        # Paper-facing rolling low/elevated diagnostic state.
        pred_df = add_uncertainty_diagnostic_state(pred_df, results_so_far, cfg)

        pred_df = compute_fallback_design_predictions(pred_df, cfg)
        pred_df["safe_var_pred"] = pred_df[safe_col]
        pred_df["fallback_design"] = fallback_design
        pred_df["use_simple_hist_fallback"] = int(fallback_design != "raw_model")
        pred_df["use_uncertainty_fallback"] = int(fallback_design in {"uncertainty_only_fallback", "full_service"})
        pred_df["use_quality_fallback"] = int(fallback_design in {"quality_only_fallback", "full_service"})

        pred_df["fallback_size"] = (pred_df["var_pred"] - pred_df["safe_var_pred"]).clip(lower=0.0)
        pred_df["fallback_ratio"] = (
            pred_df["fallback_size"] / pred_df["adj_scale"].replace(0, np.nan)
        ).fillna(0.0).clip(0.0, 1.0)

        pred_df["is_evaluable"] = pred_df[cfg.target_col].notna().astype(int)
        pred_df["breach_model"] = np.where(
            pred_df["is_evaluable"] == 1,
            (pred_df[cfg.target_col] < pred_df["var_pred"]).astype(float),
            np.nan,
        )
        pred_df["breach_safe"] = np.where(
            pred_df["is_evaluable"] == 1,
            (pred_df[cfg.target_col] < pred_df["safe_var_pred"]).astype(float),
            np.nan,
        )
        pred_df["alert_level"] = pred_df.apply(
            lambda row: alert_from_row(row, cfg, use_quality_alert=use_quality_alert),
            axis=1
        )
        pred_df["experiment"] = experiment_name
        pred_df["use_quality_feature"] = int(use_quality_feature)
        pred_df["use_quality_alert"] = int(use_quality_alert)
        pred_df["corrupt_prediction_inputs"] = int(corrupt_prediction_inputs)
        pred_df["drift_reference_name"] = drift_reference_name

        keep_cols = [
            "experiment", "fallback_design", "date", "symbol", cfg.target_col, "is_evaluable",
            "var_pred_raw", "calib_offset", "var_pred", "safe_var_pred",
            "raw_model_pred", "simple_fallback_pred", "uncertainty_only_pred", "quality_only_pred",
            "full_fallback_pred",
            "safe_hist_anchor", "safe_hist_var", "hist_var_alpha", "ewma_norm_var",
            "quality_score", "quality_flag",
            "uncertainty_score", "uncertainty_state", "uncertainty_cutoff_u", "uncertainty_history_n",
            "uncertainty_flag_alert_legacy",
            "u_model", "u_model_scaled", "u_ood", "u_drift",
            "uncertainty_adj", "quality_adj", "full_adj", "adj_scale",
            "fallback_size", "fallback_ratio",
            "breach_model", "breach_safe",
            "alert_level",
            "curve_available", "vix_available",
            "vix", "zc_30d", "zc_365d", "term_spread_1y_1m",
            "corruption_applied", "corruption_mode",
            "use_quality_feature", "use_quality_alert", "use_simple_hist_fallback",
            "use_uncertainty_fallback", "use_quality_fallback", "corrupt_prediction_inputs",
            "drift_reference_name",
        ]
        keep_cols = [c for c in keep_cols if c in pred_df.columns]

        rows_out.append(pred_df[keep_cols].copy())
        results_so_far = pd.concat(rows_out, ignore_index=True)

    if not rows_out:
        raise RuntimeError(f"Backtest produced no predictions for experiment={experiment_name}.")

    results = pd.concat(rows_out, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    summary = evaluate_results(results, cfg)
    summary["experiment"] = experiment_name
    summary["fallback_design"] = fallback_design
    summary["drift_reference_name"] = drift_reference_name
    return results, summary


# ============================================================
# Evaluation
# ============================================================

def evaluate_results(results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    eval_results = results[results["is_evaluable"] == 1].copy()

    def add_eval(name: str, y: np.ndarray, q: np.ndarray, breach: np.ndarray):
        k = kupiec_test(breach, cfg.alpha)
        rows.append({
            "segment": name,
            "n": k["n"],
            "breaches": k["x"],
            "breach_rate": k["empirical_rate"],
            "kupiec_lr_uc": k["lr_uc"],
            "pinball_loss": pinball_loss(y, q, cfg.alpha),
        })

    if eval_results.empty:
        return pd.DataFrame([{
            "segment": "no_evaluable_rows",
            "n": 0,
            "breaches": np.nan,
            "breach_rate": np.nan,
            "kupiec_lr_uc": np.nan,
            "pinball_loss": np.nan,
        }])

    y = eval_results[cfg.target_col].to_numpy()
    add_eval("overall_model_var", y, eval_results["var_pred"].to_numpy(), eval_results["breach_model"].to_numpy())
    add_eval("overall_safe_var", y, eval_results["safe_var_pred"].to_numpy(), eval_results["breach_safe"].to_numpy())

    if "safe_hist_anchor" in eval_results.columns:
        hist63_breach = (eval_results[cfg.target_col] < eval_results["safe_hist_anchor"]).astype(float).to_numpy()
        add_eval("overall_hist63_anchor", y, eval_results["safe_hist_anchor"].to_numpy(), hist63_breach)

    if "hist_var_alpha" in eval_results.columns:
        hist_breach = (eval_results[cfg.target_col] < eval_results["hist_var_alpha"]).astype(float).to_numpy()
        add_eval("overall_hist_var", y, eval_results["hist_var_alpha"].to_numpy(), hist_breach)

    if "ewma_norm_var" in eval_results.columns:
        ewma_breach = (eval_results[cfg.target_col] < eval_results["ewma_norm_var"]).astype(float).to_numpy()
        add_eval("overall_ewma_norm_var", y, eval_results["ewma_norm_var"].to_numpy(), ewma_breach)

    if "vix" in eval_results.columns and eval_results["vix"].notna().any():
        vix_cut = eval_results["vix"].quantile(0.80)
        stress = eval_results["vix"] >= vix_cut
        if stress.sum() > 0:
            sub = eval_results[stress]
            add_eval("stress_model_var", sub[cfg.target_col].to_numpy(), sub["var_pred"].to_numpy(), sub["breach_model"].to_numpy())
            add_eval("stress_safe_var", sub[cfg.target_col].to_numpy(), sub["safe_var_pred"].to_numpy(), sub["breach_safe"].to_numpy())

    for state in ["low", "elevated"]:
        if "uncertainty_state" not in eval_results.columns:
            break
        sub = eval_results[eval_results["uncertainty_state"] == state]
        if len(sub) > 0:
            add_eval(f"uncertainty_{state}_safe_var", sub[cfg.target_col].to_numpy(), sub["safe_var_pred"].to_numpy(), sub["breach_safe"].to_numpy())

    for sym, sub in eval_results.groupby("symbol"):
        add_eval(f"{sym}_safe_var", sub[cfg.target_col].to_numpy(), sub["safe_var_pred"].to_numpy(), sub["breach_safe"].to_numpy())

    alert_counts = results["alert_level"].value_counts(dropna=False).to_dict()
    rows.append({
        "segment": "alert_counts",
        "n": len(results),
        "breaches": np.nan,
        "breach_rate": np.nan,
        "kupiec_lr_uc": np.nan,
        "pinball_loss": np.nan,
        **{f"alert_{k}": v for k, v in alert_counts.items()},
    })

    coverage_counts = {
        "curve_available_rate": float(results["curve_available"].mean()) if "curve_available" in results.columns else np.nan,
        "evaluable_rate": float(results["is_evaluable"].mean()) if "is_evaluable" in results.columns else np.nan,
    }
    rows.append({
        "segment": "service_coverage",
        "n": len(results),
        "breaches": np.nan,
        "breach_rate": np.nan,
        "kupiec_lr_uc": np.nan,
        "pinball_loss": np.nan,
        **coverage_counts,
    })

    quality_counts = results["quality_flag"].value_counts(dropna=False).to_dict()
    rows.append({
        "segment": "quality_activation",
        "n": len(results),
        "breaches": np.nan,
        "breach_rate": np.nan,
        "kupiec_lr_uc": np.nan,
        "pinball_loss": np.nan,
        "mean_quality_score": float(results["quality_score"].mean()),
        "quality_green": int(quality_counts.get("green", 0)),
        "quality_yellow": int(quality_counts.get("yellow", 0)),
        "quality_red": int(quality_counts.get("red", 0)),
    })

    state_counts = results["uncertainty_state"].value_counts(dropna=False).to_dict()
    rows.append({
        "segment": "uncertainty_activation",
        "n": len(results),
        "breaches": np.nan,
        "breach_rate": np.nan,
        "kupiec_lr_uc": np.nan,
        "pinball_loss": np.nan,
        "mean_uncertainty_score": float(results["uncertainty_score"].mean()),
        "uncertainty_low": int(state_counts.get("low", 0)),
        "uncertainty_elevated": int(state_counts.get("elevated", 0)),
    })

    if "corruption_applied" in results.columns:
        corr_counts = results["corruption_mode"].value_counts(dropna=False).to_dict()
        rows.append({
            "segment": "corruption_stats",
            "n": len(results),
            "breaches": np.nan,
            "breach_rate": np.nan,
            "kupiec_lr_uc": np.nan,
            "pinball_loss": np.nan,
            "corruption_rate": float(results["corruption_applied"].mean()),
            "corrupt_missing": int(corr_counts.get("missing", 0)),
            "corrupt_stale": int(corr_counts.get("stale", 0)),
            "corrupt_ohlc": int(corr_counts.get("ohlc", 0)),
        })

    return pd.DataFrame(rows)


def make_component_key_table(summary_all: pd.DataFrame) -> pd.DataFrame:
    keep_segments = {
        "overall_safe_var",
        "overall_hist63_anchor",
        "stress_safe_var",
        "quality_activation",
        "uncertainty_activation",
        "corruption_stats",
        "service_coverage",
        "alert_counts",
    }
    out = summary_all[summary_all["segment"].isin(keep_segments)].copy()
    return out.sort_values(["experiment", "segment"]).reset_index(drop=True)


def make_quality_validation_key_table(summary_all: pd.DataFrame) -> pd.DataFrame:
    experiments = [
        "corrupt_full_service",
        "corrupt_no_quality_feature",
        "corrupt_no_quality_service_layer",
    ]
    rows = []
    for exp in experiments:
        sub = summary_all[summary_all["experiment"] == exp]
        if sub.empty:
            continue

        def get_row(segment: str) -> pd.Series:
            hit = sub[sub["segment"] == segment]
            return hit.iloc[0] if len(hit) else pd.Series(dtype=float)

        overall = get_row("overall_safe_var")
        stress = get_row("stress_safe_var")
        alerts = get_row("alert_counts")

        rows.append({
            "experiment": exp,
            "overall_breach_rate": overall.get("breach_rate", np.nan),
            "stress_breach_rate": stress.get("breach_rate", np.nan),
            "overall_pinball_loss": overall.get("pinball_loss", np.nan),
            "alert_green": alerts.get("alert_green", np.nan),
            "alert_orange": alerts.get("alert_orange", np.nan),
            "alert_red": alerts.get("alert_red", np.nan),
        })

    return pd.DataFrame(rows)


# ============================================================
# Save outputs
# ============================================================

def save_outputs(results: pd.DataFrame, summary: pd.DataFrame, cfg: Config, panel: pd.DataFrame) -> Path:
    out_dir = ensure_dir(cfg.base_dir / cfg.output_dir_name)

    results_path = out_dir / "daily_service_outputs.csv"
    summary_path = out_dir / "evaluation_summary.csv"
    cfg_path = out_dir / "run_config.json"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "full_panel_date_min": str(panel["date"].min().date()),
        "full_panel_date_max": str(panel["date"].max().date()),
        "prediction_date_min": str(results["date"].min().date()),
        "prediction_date_max": str(results["date"].max().date()),
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "n_rows_after_filtering": int(len(panel)),
        "n_predictions": int(len(results)),
        "n_evaluable_predictions": int(results["is_evaluable"].sum()) if "is_evaluable" in results.columns else int(len(results)),
        "curve_available_rate_in_predictions": float(results["curve_available"].mean()) if "curve_available" in results.columns else None,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    latest = (
        results.sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values(["alert_level", "symbol"])
    )
    latest.to_csv(out_dir / "latest_service_snapshot.csv", index=False)

    monthly = results.copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
    monthly_summary = (
        monthly.groupby(["month", "symbol"], as_index=False)
        .agg(
            avg_var_pred=("var_pred", "mean"),
            avg_safe_var_pred=("safe_var_pred", "mean"),
            avg_quality_score=("quality_score", "mean"),
            avg_uncertainty_score=("uncertainty_score", "mean"),
            avg_fallback_ratio=("fallback_ratio", "mean"),
            breach_rate_model=("breach_model", "mean"),
            breach_rate_safe=("breach_safe", "mean"),
            red_alerts=("alert_level", lambda s: int((s == "red").sum())),
            orange_alerts=("alert_level", lambda s: int((s == "orange").sum())),
            evaluable_days=("is_evaluable", "sum"),
        )
    )
    monthly_summary.to_csv(out_dir / "monthly_service_summary.csv", index=False)

    return out_dir


def save_component_experiment_outputs(
    out_dir: Path,
    component_summaries: pd.DataFrame,
    component_key_table: pd.DataFrame,
    component_daily_outputs: Optional[pd.DataFrame],
    cfg: Config,
) -> None:
    component_summaries.to_csv(out_dir / "fallback_component_summary.csv", index=False)
    component_key_table.to_csv(out_dir / "fallback_component_key_metrics.csv", index=False)

    if cfg.save_component_experiment_daily_outputs and component_daily_outputs is not None:
        component_daily_outputs.to_csv(out_dir / "fallback_component_daily_outputs.csv", index=False)


def save_quality_validation_outputs(
    out_dir: Path,
    quality_summaries: pd.DataFrame,
    quality_key_table: pd.DataFrame,
    quality_daily_outputs: Optional[pd.DataFrame],
    cfg: Config,
) -> None:
    quality_summaries.to_csv(out_dir / "quality_validation_summary.csv", index=False)
    quality_key_table.to_csv(out_dir / "quality_validation_key_metrics.csv", index=False)

    if cfg.save_quality_validation_daily_outputs and quality_daily_outputs is not None:
        quality_daily_outputs.to_csv(out_dir / "quality_validation_daily_outputs.csv", index=False)


# ============================================================
# Fallback/component experiments
# ============================================================

def run_fallback_component_experiment(
    panel: pd.DataFrame,
    cfg: Config,
    corrupt_prediction_inputs: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    prefix = "corrupt_" if corrupt_prediction_inputs else ""
    reference_design = cfg.drift_reference_design
    reference_experiment_name = f"{prefix}{reference_design}"

    exps = [
        {
            "experiment_name": f"{prefix}raw_model",
            "fallback_design": "raw_model",
            "use_quality_feature": True,
            "corrupt_prediction_inputs": corrupt_prediction_inputs,
        },
        {
            "experiment_name": f"{prefix}simple_fallback_only",
            "fallback_design": "simple_fallback_only",
            "use_quality_feature": True,
            "corrupt_prediction_inputs": corrupt_prediction_inputs,
        },
        {
            "experiment_name": f"{prefix}quality_only_fallback",
            "fallback_design": "quality_only_fallback",
            "use_quality_feature": True,
            "corrupt_prediction_inputs": corrupt_prediction_inputs,
        },
        {
            "experiment_name": f"{prefix}uncertainty_only_fallback",
            "fallback_design": "uncertainty_only_fallback",
            "use_quality_feature": True,
            "corrupt_prediction_inputs": corrupt_prediction_inputs,
        },
        {
            "experiment_name": f"{prefix}full_service",
            "fallback_design": "full_service",
            "use_quality_feature": True,
            "corrupt_prediction_inputs": corrupt_prediction_inputs,
        },
    ]

    if reference_design not in {spec["fallback_design"] for spec in exps}:
        raise ValueError(f"Unsupported drift_reference_design={reference_design!r}.")

    summary_frames = []
    daily_frames = []

    reference_results, reference_summary = run_backtest(
        panel,
        cfg,
        experiment_name=reference_experiment_name,
        fallback_design=reference_design,
        use_quality_feature=True,
        corrupt_prediction_inputs=corrupt_prediction_inputs,
        drift_reference_history=None,
        drift_reference_name=reference_experiment_name,
    )

    stored_reference = False
    for spec in exps:
        if spec["fallback_design"] == reference_design:
            results_i, summary_i = reference_results, reference_summary.copy()
            stored_reference = True
        else:
            results_i, summary_i = run_backtest(
                panel,
                cfg,
                experiment_name=spec["experiment_name"],
                fallback_design=spec["fallback_design"],
                use_quality_feature=spec["use_quality_feature"],
                corrupt_prediction_inputs=spec["corrupt_prediction_inputs"],
                drift_reference_history=reference_results,
                drift_reference_name=reference_experiment_name,
            )

        summary_frames.append(summary_i)
        if cfg.save_component_experiment_daily_outputs:
            daily_frames.append(results_i)

    if not stored_reference:
        raise RuntimeError("Reference design was not included in component experiment list.")

    component_summaries = pd.concat(summary_frames, ignore_index=True)
    component_key_table = make_component_key_table(component_summaries)
    component_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else None
    return component_summaries, component_key_table, component_daily


def run_quality_validation_experiment(
    panel: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Focused corrupted-input quality-validation ablation used by the paper.

    Compares:
    - corrupt_full_service
    - corrupt_no_quality_feature
    - corrupt_no_quality_service_layer

    The drift reference is kept fixed to the corrupt_full_service run so that
    the ablations are compared under a common corrupted-input service baseline.
    """
    reference_results, reference_summary = run_backtest(
        panel,
        cfg,
        experiment_name="corrupt_full_service",
        fallback_design="full_service",
        use_quality_feature=True,
        use_quality_alert=True,
        corrupt_prediction_inputs=True,
        drift_reference_history=None,
        drift_reference_name="corrupt_full_service",
    )

    summary_frames = [reference_summary.copy()]
    daily_frames = [reference_results] if cfg.save_quality_validation_daily_outputs else []

    no_q_feature_results, no_q_feature_summary = run_backtest(
        panel,
        cfg,
        experiment_name="corrupt_no_quality_feature",
        fallback_design="full_service",
        use_quality_feature=False,
        use_quality_alert=True,
        corrupt_prediction_inputs=True,
        drift_reference_history=reference_results,
        drift_reference_name="corrupt_full_service",
    )
    summary_frames.append(no_q_feature_summary)
    if cfg.save_quality_validation_daily_outputs:
        daily_frames.append(no_q_feature_results)

    no_q_service_results, no_q_service_summary = run_backtest(
        panel,
        cfg,
        experiment_name="corrupt_no_quality_service_layer",
        fallback_design="uncertainty_only_fallback",
        use_quality_feature=True,
        use_quality_alert=False,
        corrupt_prediction_inputs=True,
        drift_reference_history=reference_results,
        drift_reference_name="corrupt_full_service",
    )
    summary_frames.append(no_q_service_summary)
    if cfg.save_quality_validation_daily_outputs:
        daily_frames.append(no_q_service_results)

    quality_summaries = pd.concat(summary_frames, ignore_index=True)
    quality_key_table = make_quality_validation_key_table(quality_summaries)
    quality_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else None
    return quality_summaries, quality_key_table, quality_daily


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()
    panel = build_dataset(cfg)

    # Main service backtest (paper default)
    results, summary = run_backtest(
        panel,
        cfg,
        experiment_name="main_full_service",
        fallback_design="full_service",
        use_quality_feature=True,
        corrupt_prediction_inputs=False,
    )
    out_dir = save_outputs(results, summary, cfg, panel)

    print("=" * 100)
    print("MAIN SERVICE BACKTEST")
    print("=" * 100)
    print(f"BASE_DIR : {cfg.base_dir}")
    print(f"OUT_DIR  : {out_dir}")
    print(f"Full panel span : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Prediction span : {results['date'].min().date()} -> {results['date'].max().date()}")
    print(f"Symbols         : {sorted(panel['symbol'].unique().tolist())}")
    print(f"Rows            : {len(panel):,}")
    print(f"Pred rows       : {len(results):,}")
    print(f"Evaluable rows  : {int(results['is_evaluable'].sum()):,}")
    print("-" * 100)
    print(summary.to_string(index=False))

    component_frames = []
    component_daily_frames = []

    if cfg.run_component_ablation_experiment:
        print("=" * 100)
        print("FALLBACK COMPONENT EXPERIMENT (CLEAN INPUTS)")
        print("=" * 100)
        comp_summary, comp_key, comp_daily = run_fallback_component_experiment(
            panel, cfg, corrupt_prediction_inputs=False
        )
        component_frames.append(comp_summary)
        if comp_daily is not None:
            component_daily_frames.append(comp_daily)
        print(comp_key.to_string(index=False))

    if cfg.run_corrupted_component_experiment:
        print("=" * 100)
        print("FALLBACK COMPONENT EXPERIMENT (CORRUPTED INPUTS)")
        print("=" * 100)
        comp_summary_c, comp_key_c, comp_daily_c = run_fallback_component_experiment(
            panel, cfg, corrupt_prediction_inputs=True
        )
        component_frames.append(comp_summary_c)
        if comp_daily_c is not None:
            component_daily_frames.append(comp_daily_c)
        print(comp_key_c.to_string(index=False))

    if component_frames:
        component_summaries = pd.concat(component_frames, ignore_index=True)
        component_key_table = make_component_key_table(component_summaries)
        component_daily = pd.concat(component_daily_frames, ignore_index=True) if component_daily_frames else None
        save_component_experiment_outputs(out_dir, component_summaries, component_key_table, component_daily, cfg)

    if cfg.run_quality_validation_experiment:
        print("=" * 100)
        print("FOCUSED QUALITY-VALIDATION ABLATION (CORRUPTED INPUTS)")
        print("=" * 100)
        q_summary, q_key, q_daily = run_quality_validation_experiment(panel, cfg)
        save_quality_validation_outputs(out_dir, q_summary, q_key, q_daily, cfg)
        print(q_key.to_string(index=False))

    print("=" * 100)
    print("Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
