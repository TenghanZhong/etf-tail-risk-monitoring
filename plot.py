from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE_DIR = Path(r"C:\Users\26876\Desktop\2026BIgdataservice\risk_monitoring_service_outputs_revised3")
DAILY_CSV = BASE_DIR / "daily_service_outputs.csv"
MONTHLY_CSV = BASE_DIR / "monthly_service_summary.csv"
FIG_DIR = BASE_DIR / "figures"
ROLLING_WINDOW = 60
STRESS_VIX_QUANTILE = 0.80
DPI = 400

def sparse_month_ticks(labels, step=2, short=True):
    idx = np.arange(len(labels))
    show = idx[::step]
    shown_labels = []

    for i in show:
        lab = labels[i]
        if short:
            lab = lab[2:] if len(lab) >= 5 else lab   # 2023-01 -> 23-01
        shown_labels.append(lab)
    return show, shown_labels
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_pipeline_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.36, 0.16, 0.34, "Inputs\nETF panel, VIX,\nyield curve"),
        (0.215, 0.36, 0.16, 0.34, "Quality control\nmissingness, OHLC,\njumps, stale"),
        (0.41, 0.36, 0.16, 0.34, "Tail-risk model\n5% quantile GBDT\n+ calibration"),
        (0.605, 0.36, 0.16, 0.34, "Uncertainty\ndispersion, OOD,\ndrift"),
        (0.80, 0.36, 0.16, 0.34, "Safe output\nhistorical VaR,\nfallback, alerts"),
    ]

    for (x, y, w, h, text) in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.6,
            facecolor="#f7f7f7",
            edgecolor="black",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=12.5, linespacing=1.15
        )

    out_box = FancyBboxPatch(
        (0.28, 0.07), 0.44, 0.16,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        facecolor="#eef3ff",
        edgecolor="black",
    )
    ax.add_patch(out_box)
    ax.text(
        0.50, 0.15,
        "Service outputs: daily snapshots, alerts, summaries, diagnostics",
        ha="center", va="center",
        fontsize=12.2
    )

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        ax.annotate(
            "",
            xy=(x1 - 0.008, y),
            xytext=(x0 + 0.008, y),
            arrowprops=dict(arrowstyle="->", lw=2.0, shrinkA=0, shrinkB=0),
        )

    ax.annotate(
        "",
        xy=(0.50, 0.23),
        xytext=(0.50, 0.36),
        arrowprops=dict(arrowstyle="->", lw=2.0, shrinkA=0, shrinkB=0),
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.96, bottom=0.05)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_rolling_breach_rates(daily: pd.DataFrame, out_path: Path, window: int = 60) -> None:
    import matplotlib.dates as mdates

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_evaluable"] == 1].copy()
    df["breach_hist"] = (df["ret_next"] < df["hist_var_alpha"]).astype(int)

    by_date = (
        df.groupby("date")[["breach_model", "breach_safe", "breach_hist"]]
        .mean()
        .sort_index()
    )
    roll = by_date.rolling(window=window, min_periods=max(15, window // 3)).mean()

    fig, ax = plt.subplots(figsize=(7.2, 3.5))   # 单栏友好；若双栏可改成 (10.5, 3.8)

    ax.plot(
        roll.index, roll["breach_model"],
        linewidth=2.0, label="Model VaR"
    )
    ax.plot(
        roll.index, roll["breach_safe"],
        linewidth=2.2, label="Safe VaR"
    )
    ax.plot(
        roll.index, roll["breach_hist"],
        linewidth=2.0, label="Historical VaR"
    )
    ax.axhline(
        0.05,
        linestyle="--",
        linewidth=1.6,
        label="Target 5%"
    )

    ax.set_ylabel("Breach rate", fontsize=10.5)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=9)

    # 时间轴：每季度一个刻度，避免太密
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 更紧的 y 轴，提高可读性
    ymax = np.nanmax(
        [
            roll["breach_model"].max(),
            roll["breach_safe"].max(),
            roll["breach_hist"].max(),
            0.05,
        ]
    )
    ax.set_ylim(0, max(0.10, ymax + 0.01))

    # 只保留 y 方向网格
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    # 图例不要横向铺满一行
    ax.legend(
        frameon=False,
        fontsize=8.8,
        ncol=2,
        loc="upper left"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_stress_comparison(daily: pd.DataFrame, out_path: Path, q: float = 0.80) -> None:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_evaluable"] == 1].copy()
    df["breach_hist"] = (df["ret_next"] < df["hist_var_alpha"]).astype(int)

    vix_cut = df["vix"].quantile(q)
    df["regime"] = np.where(df["vix"] >= vix_cut, "Stress", "Non-stress")

    comp = df.groupby("regime")[["breach_model", "breach_safe", "breach_hist"]].mean()
    comp = comp.reindex(["Non-stress", "Stress"])

    x = np.arange(len(comp.index))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    b1 = ax.bar(x - width, comp["breach_model"].values, width, label="Model VaR")
    b2 = ax.bar(x,         comp["breach_safe"].values, width, label="Safe VaR")
    b3 = ax.bar(x + width, comp["breach_hist"].values, width, label="Historical VaR")

    ax.axhline(0.05, linestyle="--", linewidth=1.4, label="Target 5%")

    ax.set_xticks(x)
    ax.set_xticklabels(comp.index)
    ax.set_ylabel("Breach rate")

    # 给顶部留足空间，避免数字撞边框/图例
    ymax = max(
        comp["breach_model"].max(),
        comp["breach_safe"].max(),
        comp["breach_hist"].max(),
        0.05
    )
    ax.set_ylim(0, ymax + 0.012)

    # 图例放右上角图内，避免和数字重叠
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    # 柱顶数值标签
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.0025,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5
            )

    # 不要在图里再加 Figure 3 标题，标题交给 LaTeX caption
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_stress_comparison(daily: pd.DataFrame, out_path: Path, q: float = 0.80) -> None:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_evaluable"] == 1].copy()
    df["breach_hist"] = (df["ret_next"] < df["hist_var_alpha"]).astype(int)

    vix_cut = df["vix"].quantile(q)
    df["regime"] = np.where(df["vix"] >= vix_cut, "Stress", "Non-stress")

    comp = df.groupby("regime")[["breach_model", "breach_safe", "breach_hist"]].mean()
    comp = comp.reindex(["Non-stress", "Stress"])

    x = np.arange(len(comp.index))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    b1 = ax.bar(x - width, comp["breach_model"].values, width, label="Model VaR")
    b2 = ax.bar(x,         comp["breach_safe"].values, width, label="Safe VaR")
    b3 = ax.bar(x + width, comp["breach_hist"].values, width, label="Historical VaR")

    ax.axhline(0.05, linestyle="--", linewidth=1.4, label="Target 5%")

    ax.set_xticks(x)
    ax.set_xticklabels(comp.index)
    ax.set_ylabel("Breach rate")

    # 给顶部留足空间，避免数字撞边框/图例
    ymax = max(
        comp["breach_model"].max(),
        comp["breach_safe"].max(),
        comp["breach_hist"].max(),
        0.05
    )
    ax.set_ylim(0, ymax + 0.012)

    # 图例放右上角图内，避免和数字重叠
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    # 柱顶数值标签
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.0025,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5
            )

    # 不要在图里再加 Figure 3 标题，标题交给 LaTeX caption
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_monthly_alerts(daily: pd.DataFrame, out_path: Path) -> None:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    alert_counts = (
        df.groupby(["month", "alert_level"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    for col in ["green", "orange", "red"]:
        if col not in alert_counts.columns:
            alert_counts[col] = 0
    alert_counts = alert_counts[["green", "orange", "red"]]

    # 100% stacked shares
    totals = alert_counts.sum(axis=1).replace(0, np.nan)
    alert_share = alert_counts.div(totals, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(alert_share.index))
    width = 0.82

    ax.bar(
        x,
        alert_share["green"],
        width=width,
        label="Green",
        color="#4C78A8",
        linewidth=0,
    )
    ax.bar(
        x,
        alert_share["orange"],
        width=width,
        bottom=alert_share["green"],
        label="Orange",
        color="#F58518",
        linewidth=0,
    )
    ax.bar(
        x,
        alert_share["red"],
        width=width,
        bottom=alert_share["green"] + alert_share["orange"],
        label="Red",
        color="#54A24B",
        linewidth=0,
    )

    tick_idx, tick_labels = sparse_month_ticks(
        alert_share.index.tolist(), step=3, short=True
    )
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    ax.set_ylabel("Share", fontsize=9.5)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels(
        [f"{int(v * 100)}%" for v in np.linspace(0, 1.0, 6)],
        fontsize=8.5
    )

    # Smaller legend, moved upward
    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=7.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=0.9,
        borderaxespad=0.2,
        labelspacing=0.3,
    )

    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.margins(x=0.01)

    # Leave explicit room for legend above the axes
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.08, right=0.99)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_uncertainty_heatmap(monthly: pd.DataFrame, out_path: Path) -> None:
    mon = monthly.copy()
    mon["month"] = mon["month"].astype(str)

    heat = mon.pivot_table(
        index="symbol",
        columns="month",
        values="avg_uncertainty_score",
        aggfunc="mean",
    )
    heat = heat.reindex(sorted(heat.index), axis=0)
    heat = heat.reindex(sorted(heat.columns), axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    im = ax.imshow(heat.values, aspect="auto")

    tick_idx, tick_labels = sparse_month_ticks(heat.columns.tolist(), step=2, short=True)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8.5)

    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=9.5)
    ax.set_title("Month-Symbol Uncertainty Heatmap", fontsize=11.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Avg uncertainty score", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def main() -> None:
    ensure_dir(FIG_DIR)

    daily = pd.read_csv(DAILY_CSV)
    monthly = pd.read_csv(MONTHLY_CSV)

    plot_pipeline_diagram(FIG_DIR / "figure1_pipeline_diagram.png")
    plot_rolling_breach_rates(daily, FIG_DIR / "figure2_rolling_breach_rates.png", window=ROLLING_WINDOW)
    plot_stress_comparison(daily, FIG_DIR / "figure3_stress_vs_nonstress.png", q=STRESS_VIX_QUANTILE)
    plot_monthly_alerts(daily, FIG_DIR / "figure4a_monthly_alerts.png")
    plot_uncertainty_heatmap(monthly, FIG_DIR / "figure4b_uncertainty_heatmap.png")

    print("=" * 90)
    print("Figures saved to:", FIG_DIR)
    print("1) figure1_pipeline_diagram.png")
    print("2) figure2_rolling_breach_rates.png")
    print("3) figure3_stress_vs_nonstress.png")
    print("4) figure4a_monthly_alerts.png")
    print("5) figure4b_uncertainty_heatmap.png")
    print("=" * 90)


if __name__ == "__main__":
    main()
