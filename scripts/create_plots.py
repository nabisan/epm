import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

print("Loading data...")
predictions = pd.read_csv("/workspace/data/processed/epm_predictions_top50.csv")
backtest = pd.read_csv("/workspace/data/processed/backtest_results_top50.csv")

sns.set_style("whitegrid")
output_dir = Path("/workspace/data/processed")

print(f"Predictions: {len(predictions)} tickers")
print(f"Backtest: {len(backtest)} tickers")

# Plot 1: Quintile Win Rate
print("\nCreating Plot 1: Quintile Win Rate...")
fig, ax = plt.subplots(figsize=(12, 6))
quintile_win = backtest.groupby("quintile")["epm_better"].agg(["mean", "count"]).reset_index()
quintile_win["win_pct"] = quintile_win["mean"] * 100

bars = ax.bar(
    range(len(quintile_win)), 
    quintile_win["win_pct"], 
    color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"],
    edgecolor="black", 
    linewidth=2, 
    alpha=0.8
)
ax.axhline(y=50, color="red", linestyle="--", linewidth=2, label="Random (50%)")
ax.set_xticks(range(len(quintile_win)))
ax.set_xticklabels(quintile_win["quintile"], rotation=15, ha="right")
ax.set_ylabel("Win Rate (%)", fontsize=13, fontweight="bold")
ax.set_title("EPM Win Rate by Quintile (n=47)", fontsize=15, fontweight="bold")
ax.set_ylim([0, 100])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

for bar, row in zip(bars, quintile_win.itertuples()):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2., 
        height + 2,
        f"{height:.1f}%\n(n={row.count})",
        ha="center", 
        va="bottom", 
        fontsize=10, 
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(output_dir / "quintile_win_rate.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_dir}/quintile_win_rate.png")
plt.close()

# Plot 2: Sector Performance
print("Creating Plot 2: Sector Performance...")
fig, ax = plt.subplots(figsize=(12, 8))
sector_perf = backtest.groupby("sector").agg({
    "epm_better": "mean",
    "ticker": "count"
}).sort_values("epm_better", ascending=True)
sector_perf["win_pct"] = sector_perf["epm_better"] * 100

colors = ["green" if x > 50 else "red" for x in sector_perf["win_pct"]]
bars = ax.barh(
    range(len(sector_perf)), 
    sector_perf["win_pct"], 
    color=colors, 
    edgecolor="black", 
    linewidth=1.5, 
    alpha=0.7
)
ax.axvline(x=50, color="black", linestyle="--", linewidth=2, label="Random (50%)")
ax.set_yticks(range(len(sector_perf)))
ax.set_yticklabels(sector_perf.index)
ax.set_xlabel("Win Rate (%)", fontsize=13, fontweight="bold")
ax.set_title("EPM Win Rate by Sector", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="x")

for i, (bar, idx) in enumerate(zip(bars, sector_perf.index)):
    width = bar.get_width()
    count = sector_perf.loc[idx, "ticker"]
    ax.text(
        width + 2, 
        bar.get_y() + bar.get_height()/2.,
        f"{width:.1f}% (n={int(count)})",
        ha="left", 
        va="center", 
        fontsize=9, 
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(output_dir / "sector_performance.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_dir}/sector_performance.png")
plt.close()

# Plot 3: Adjustment Distribution
print("Creating Plot 3: Adjustment Distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(
    predictions["adjustment_pct"], 
    bins=20, 
    edgecolor="black", 
    alpha=0.7, 
    color="steelblue"
)
ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No Adjustment")
ax.axvline(
    x=predictions["adjustment_pct"].mean(), 
    color="green", 
    linestyle="--", 
    linewidth=2, 
    label=f"Mean: {predictions[adjustment_pct].mean():.2f}%"
)
ax.set_xlabel("EPM Adjustment (%)", fontsize=13, fontweight="bold")
ax.set_ylabel("Frequency", fontsize=13, fontweight="bold")
ax.set_title("Distribution of EPM Adjustments (n=49)", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "adjustment_distribution.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_dir}/adjustment_distribution.png")
plt.close()

print("\n" + "="*60)
print("ALL PLOTS CREATED SUCCESSFULLY!")
print("="*60)
print(f"Location: {output_dir}")
print("\nFiles created:")
print("  1. quintile_win_rate.png")
print("  2. sector_performance.png")
print("  3. adjustment_distribution.png")
print("="*60)
