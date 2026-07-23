"""Figure-generation utilities for ETF tail-risk monitoring results."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


BASE_DIR = Path(
    os.environ.get("ETF_TAIL_RISK_DATA_DIR", Path(__file__).resolve().parent)
)
DAILY_CSV = BASE_DIR / "daily_service_outputs.csv"
MONTHLY_CSV = BASE_DIR / "monthly_service_summary.csv"
FIG_DIR = BASE_DIR / "figures"
ROLLING_WINDOW = 60
STRESS_VIX_QUANTILE = 0.80
DPI = 400


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sparse_month_ticks(labels: list[str], step: int = 2) -> tuple[np.ndarray, list[str]]:
    indices = np.arange(len(labels))
    selected = indices[::step]
    short_labels = [labels[i][2:] if len(labels[i]) >= 5 else labels[i] for i in selected]
    return selected, short_labels


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

    for x, y, width, height, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.6,
            facecolor="#f7f7f7",
            edgecolor="black",
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=12.5,
            linespacing=1.15,
        )

    output_box = FancyBboxPatch(
        (0.28, 0.07),
        0.44,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        facecolor="#eef3ff",
        edgecolor="black",
    )
    ax.add_patch(output_box)
    ax.text(
        0.50,
        0.15,
        "Service outputs: daily snapshots, alerts, summaries, diagnostics",
        ha="center",
        va="center",
        fontsize=12.2,
    )

    for index in range(len(boxes) - 1):
        x_start = boxes[index][0] + boxes[index][2]
        x_end = boxes[index + 1][0]
        y_mid = boxes[index][1] + boxes[index][3] / 2
        ax.annotate(
            "",
            xy=(x_end - 0.008, y_mid),
            xytext=(x_start + 0.008, y_mid),
            arrowprops={"arrowstyle": "->", "lw": 2.0, "shrinkA": 0, "shrinkB": 0},
        )

    ax.annotate(
        "",
        xy=(0.50, 0.23),
        xytext=(0.50, 0.36),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "shrinkA": 0, "shrinkB": 0},
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.96, bottom=0.05)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_rolling_breach_rates(
    daily: pd.DataFrame,
    out_path: Path,
    window: int = ROLLING_WINDOW,
) -> None:
    import matplotlib.dates as mdates

    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["is_evaluable"] == 1].copy()
    data["breach_hist"] = (data["ret_next"] < data["hist_var_alpha"]).astype(int)

    by_date = (
        data.groupby("date")[["breach_model", "breach_safe", "breach_hist"]]
        .mean()
        .sort_index()
    )
    rolling = by_date.rolling(window=window, min_periods=max(15, window // 3)).mean()

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.plot(rolling.index, rolling["breach_model"], linewidth=2.0, label="Model VaR")
    ax.plot(rolling.index, rolling["breach_safe"], linewidth=2.2, label="Safe VaR")
    ax.plot(rolling.index, rolling["breach_hist"], linewidth=2.0, label="Historical VaR")
    ax.axhline(0.05, linestyle="--", linewidth=1.6, label="Target 5%")

    ax.set_ylabel("Breach rate", fontsize=10.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    maximum = np.nanmax(
        [
            rolling["breach_model"].max(),
            rolling["breach_safe"].max(),
            rolling["breach_hist"].max(),
            0.05,
        ]
    )
    ax.set_ylim(0, max(0.10, maximum + 0.01))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.8, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_stress_comparison(
    daily: pd.DataFrame,
    out_path: Path,
    quantile: float = STRESS_VIX_QUANTILE,
) -> None:
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["is_evaluable"] == 1].copy()
    data["breach_hist"] = (data["ret_next"] < data["hist_var_alpha"]).astype(int)

    vix_cutoff = data["vix"].quantile(quantile)
    data["regime"] = np.where(data["vix"] >= vix_cutoff, "Stress", "Non-stress")
    comparison = data.groupby("regime")[["breach_model", "breach_safe", "breach_hist"]].mean()
    comparison = comparison.reindex(["Non-stress", "Stress"])

    x_values = np.arange(len(comparison.index))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    bars = [
        ax.bar(x_values - width, comparison["breach_model"], width, label="Model VaR"),
        ax.bar(x_values, comparison["breach_safe"], width, label="Safe VaR"),
        ax.bar(x_values + width, comparison["breach_hist"], width, label="Historical VaR"),
    ]
    ax.axhline(0.05, linestyle="--", linewidth=1.4, label="Target 5%")
    ax.set_xticks(x_values)
    ax.set_xticklabels(comparison.index)
    ax.set_ylabel("Breach rate")

    maximum = max(
        comparison["breach_model"].max(),
        comparison["breach_safe"].max(),
        comparison["breach_hist"].max(),
        0.05,
    )
    ax.set_ylim(0, maximum + 0.012)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.0025,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_monthly_alerts(daily: pd.DataFrame, out_path: Path) -> None:
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.to_period("M").astype(str)

    counts = data.groupby(["month", "alert_level"]).size().unstack(fill_value=0).sort_index()
    for column in ["green", "orange", "red"]:
        if column not in counts.columns:
            counts[column] = 0
    counts = counts[["green", "orange", "red"]]
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x_values = np.arange(len(shares.index))
    width = 0.82
    ax.bar(x_values, shares["green"], width=width, label="Green", linewidth=0)
    ax.bar(
        x_values,
        shares["orange"],
        width=width,
        bottom=shares["green"],
        label="Orange",
        linewidth=0,
    )
    ax.bar(
        x_values,
        shares["red"],
        width=width,
        bottom=shares["green"] + shares["orange"],
        label="Red",
        linewidth=0,
    )

    tick_indices, tick_labels = sparse_month_ticks(shares.index.tolist(), step=3)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Share", fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1.0, 6)], fontsize=8.5)
    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=7.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.01)

    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.08, right=0.99)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_uncertainty_heatmap(monthly: pd.DataFrame, out_path: Path) -> None:
    data = monthly.copy()
    data["month"] = data["month"].astype(str)
    heatmap = data.pivot_table(
        index="symbol",
        columns="month",
        values="avg_uncertainty_score",
        aggfunc="mean",
    )
    heatmap = heatmap.reindex(sorted(heatmap.index), axis=0)
    heatmap = heatmap.reindex(sorted(heatmap.columns), axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    image = ax.imshow(heatmap.values, aspect="auto")
    tick_indices, tick_labels = sparse_month_ticks(heatmap.columns.tolist(), step=2)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=9.5)
    ax.set_title("Month-Symbol Uncertainty Heatmap", fontsize=11.5)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("Average uncertainty score", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ensure_dir(FIG_DIR)
    daily = pd.read_csv(DAILY_CSV)
    monthly = pd.read_csv(MONTHLY_CSV)

    plot_pipeline_diagram(FIG_DIR / "figure1_pipeline_diagram.png")
    plot_rolling_breach_rates(daily, FIG_DIR / "figure2_rolling_breach_rates.png")
    plot_stress_comparison(daily, FIG_DIR / "figure3_stress_vs_nonstress.png")
    plot_monthly_alerts(daily, FIG_DIR / "figure4a_monthly_alerts.png")
    plot_uncertainty_heatmap(monthly, FIG_DIR / "figure4b_uncertainty_heatmap.png")

    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
