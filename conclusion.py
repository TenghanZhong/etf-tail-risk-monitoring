from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# 路径
# =========================
BASE_DIR = Path(r"C:\Users\26876\Desktop\2026BIgdataservice\risk_monitoring_service_outputs_revised2")
DAILY_CSV = BASE_DIR / "daily_service_outputs.csv"
CFG_JSON = BASE_DIR / "run_config.json"

# =========================
# 读取数据
# =========================
df = pd.read_csv(DAILY_CSV)
with open(CFG_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

cfg = meta["config"]

orange_fallback_ratio = float(cfg["orange_fallback_ratio"])
red_fallback_ratio = float(cfg["red_fallback_ratio"])
orange_drift_score = float(cfg["orange_drift_score"])
red_drift_score = float(cfg["red_drift_score"])

# =========================
# 定义触发条件
# =========================
df["hit_quality_red"] = df["quality_flag"].eq("red")
df["hit_quality_yellow"] = df["quality_flag"].eq("yellow")

df["hit_unc_high"] = df["uncertainty_flag"].eq("high")
df["hit_unc_medium"] = df["uncertainty_flag"].eq("medium")

df["hit_drift_red"] = df["u_drift"] >= red_drift_score
df["hit_drift_orange"] = df["u_drift"] >= orange_drift_score

df["hit_fallback_red"] = df["fallback_ratio"] >= red_fallback_ratio
df["hit_fallback_orange"] = df["fallback_ratio"] >= orange_fallback_ratio

# =========================
# 按代码真实优先级分配“主导来源”
# red 优先级：
# quality_red > uncertainty_high > drift_red > fallback_red
#
# orange 优先级：
# quality_yellow > uncertainty_medium > drift_orange > fallback_orange
# =========================
red_conditions = [
    df["hit_quality_red"],
    df["hit_unc_high"],
    df["hit_drift_red"],
    df["hit_fallback_red"],
]
red_choices = [
    "quality_red",
    "uncertainty_high",
    "drift_red",
    "fallback_red",
]

orange_conditions = [
    df["hit_quality_yellow"],
    df["hit_unc_medium"],
    df["hit_drift_orange"],
    df["hit_fallback_orange"],
]
orange_choices = [
    "quality_yellow",
    "uncertainty_medium",
    "drift_orange",
    "fallback_orange",
]

df["primary_red_source"] = np.where(
    df["alert_level"].eq("red"),
    np.select(red_conditions, red_choices, default="other"),
    np.nan,
)

df["primary_orange_source"] = np.where(
    df["alert_level"].eq("orange"),
    np.select(orange_conditions, orange_choices, default="other"),
    np.nan,
)

# =========================
# 1) Raw trigger hits
# 说明：同一行可同时命中多个条件，所以这一表总和可能 > alert 条数
# =========================
def build_raw_hits_table(sub: pd.DataFrame, level_name: str, hit_cols_map: dict[str, str]) -> pd.DataFrame:
    n = len(sub)
    rows = []
    for label, col in hit_cols_map.items():
        cnt = int(sub[col].sum())
        share = cnt / n if n > 0 else np.nan
        rows.append({
            "alert_level": level_name,
            "trigger": label,
            "count": cnt,
            "share_within_alert_level": share,
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)

red_raw = build_raw_hits_table(
    df[df["alert_level"] == "red"].copy(),
    "red",
    {
        "quality_red": "hit_quality_red",
        "uncertainty_high": "hit_unc_high",
        "drift_red": "hit_drift_red",
        "fallback_red": "hit_fallback_red",
    },
)

orange_raw = build_raw_hits_table(
    df[df["alert_level"] == "orange"].copy(),
    "orange",
    {
        "quality_yellow": "hit_quality_yellow",
        "uncertainty_medium": "hit_unc_medium",
        "drift_orange": "hit_drift_orange",
        "fallback_orange": "hit_fallback_orange",
    },
)

raw_hits = pd.concat([red_raw, orange_raw], ignore_index=True)

# =========================
# 2) Primary trigger decomposition
# 说明：严格一条 alert 只归一个主导来源
# =========================
def build_primary_table(sub: pd.DataFrame, level_name: str, source_col: str) -> pd.DataFrame:
    vc = sub[source_col].value_counts(dropna=False)
    n = len(sub)
    out = vc.rename_axis("primary_source").reset_index(name="count")
    out["alert_level"] = level_name
    out["share_within_alert_level"] = out["count"] / n if n > 0 else np.nan
    return out[["alert_level", "primary_source", "count", "share_within_alert_level"]]

red_primary = build_primary_table(
    df[df["alert_level"] == "red"].copy(),
    "red",
    "primary_red_source",
)

orange_primary = build_primary_table(
    df[df["alert_level"] == "orange"].copy(),
    "orange",
    "primary_orange_source",
)

primary_decomp = pd.concat([red_primary, orange_primary], ignore_index=True)

# =========================
# 3) 按月份再做一版主导来源分解
# 这个很适合后面看 dashboard 为啥长期偏红
# =========================
df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

red_monthly_primary = (
    df[df["alert_level"] == "red"]
    .groupby(["month", "primary_red_source"])
    .size()
    .reset_index(name="count")
    .rename(columns={"primary_red_source": "primary_source"})
)

orange_monthly_primary = (
    df[df["alert_level"] == "orange"]
    .groupby(["month", "primary_orange_source"])
    .size()
    .reset_index(name="count")
    .rename(columns={"primary_orange_source": "primary_source"})
)

# =========================
# 保存
# =========================
raw_hits_path = BASE_DIR / "alert_trigger_raw_hits.csv"
primary_path = BASE_DIR / "alert_trigger_primary_decomposition.csv"
red_monthly_path = BASE_DIR / "alert_trigger_red_monthly_primary.csv"
orange_monthly_path = BASE_DIR / "alert_trigger_orange_monthly_primary.csv"

raw_hits.to_csv(raw_hits_path, index=False)
primary_decomp.to_csv(primary_path, index=False)
red_monthly_primary.to_csv(red_monthly_path, index=False)
orange_monthly_primary.to_csv(orange_monthly_path, index=False)

# =========================
# 打印摘要
# =========================
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)

print("=" * 100)
print("RAW TRIGGER HITS")
print("=" * 100)
print(raw_hits.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 100)
print("PRIMARY TRIGGER DECOMPOSITION")
print("=" * 100)
print(primary_decomp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 100)
print("Saved files:")
print(raw_hits_path)
print(primary_path)
print(red_monthly_path)
print(orange_monthly_path)
print("=" * 100)