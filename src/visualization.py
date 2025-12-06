# src/visualization.py
"""
Visualization helpers for the EDA notebook.

Each plotting function returns a matplotlib Figure object so you can further customize
in the notebook, save it, or display inline.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid", context="notebook")

def hist_numeric(df: pd.DataFrame, col: str, bins: int = 50, figsize=(8,4), log_scale: bool = False):
    """Histogram + KDE for a numeric column."""
    fig, ax = plt.subplots(figsize=figsize)
    data = df[col].dropna()
    if log_scale:
        data = data[data > 0]
        ax.set_xscale("log")
    sns.histplot(data, bins=bins, kde=True, ax=ax)
    ax.set_title(f"Distribution of {col} (n={len(data)})")
    return fig

def bar_top_categories(df: pd.DataFrame, col: str, top_n: int = 10, figsize=(8,4)):
    """Bar chart for top N categories for a categorical column."""
    fig, ax = plt.subplots(figsize=figsize)
    vc = df[col].fillna("Unknown").value_counts().nlargest(top_n)
    sns.barplot(x=vc.values, y=vc.index, ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    ax.set_title(f"Top {top_n} {col}")
    return fig

def boxplot_by_category(df: pd.DataFrame, numeric_col: str, cat_col: str, figsize=(10,6), max_categories: int = 20):
    """Boxplot of a numeric column grouped by a categorical column (limited categories)."""
    fig, ax = plt.subplots(figsize=figsize)
    top_categories = df[cat_col].value_counts().index[:max_categories]
    plot_df = df[df[cat_col].isin(top_categories)]
    sns.boxplot(data=plot_df, x=cat_col, y=numeric_col, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"{numeric_col} by {cat_col} (top {len(top_categories)} categories)")
    return fig

def correlation_heatmap(corr: pd.DataFrame, figsize=(10,8), annot: bool = True, cmap: str = "vlag"):
    """Plot a heatmap for a correlation matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, center=0, ax=ax)
    ax.set_title("Correlation matrix")
    return fig

def lossratio_heatmap_by_province_vehicle(df: pd.DataFrame,
                                          province_col: str = "Province",
                                          vehicle_col: str = "VehicleType",
                                          figsize=(12, 8)):
    """
    Compute loss ratio = sum(TotalClaims)/sum(TotalPremium) pivot by Province x VehicleType and plot heatmap.
    Handles zero premiums safely.
    """
    df_ = df.copy()
    # Aggregate sums
    pivot = df_.groupby([province_col, vehicle_col]).agg(
        total_claims=("TotalClaims", "sum"),
        total_premium=("TotalPremium", "sum")
    ).reset_index()

    # Avoid division by zero
    pivot["loss_ratio"] = pivot.apply(
        lambda r: np.nan if r["total_premium"] == 0 else r["total_claims"] / r["total_premium"], axis=1
    )

    # Pivot for heatmap
    heat = pivot.pivot(index=province_col, columns=vehicle_col, values="loss_ratio")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heat, annot=True, fmt=".2f", linewidths=0.5, ax=ax, cbar_kws={"label": "Loss Ratio"}, cmap="YlGnBu")
    ax.set_title("Loss Ratio by Province (rows) and VehicleType (cols)")
    ax.set_ylabel(province_col)
    ax.set_xlabel(vehicle_col)
    plt.tight_layout()
    return fig



def timeseries_small_multiples(monthly_df: pd.DataFrame, date_col: str = "month", figsize=(12,9)):
    """
    Expect monthly_df to include columns: date_col, TotalPremium, TotalClaims, loss_ratio.
    Produces three stacked time-series plots.
    """
    fig, axes = plt.subplots(3,1, figsize=figsize, sharex=True)
    x = monthly_df[date_col].astype(str)
    axes[0].plot(x, monthly_df["TotalPremium"], marker="o")
    axes[0].set_title("Total Premium over time")
    axes[1].plot(x, monthly_df["TotalClaims"], marker="o")
    axes[1].set_title("Total Claims over time")
    axes[2].plot(x, monthly_df["loss_ratio"], marker="o")
    axes[2].set_title("Loss Ratio over time")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def joint_log_scatter_premium_claims(df: pd.DataFrame,
                                     x_col: str = "TotalPremium",
                                     y_col: str = "TotalClaims",
                                     figsize=(8, 8)):
    """
    Joint scatter of TotalPremium vs TotalClaims with log-log scaling and marginal histograms.
    Skips zero or negative values to avoid log(0) issues.
    """
    # Filter positive values for logs
    sub = df[[x_col, y_col]].dropna()
    sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]

    # Create jointgrid
    g = sns.JointGrid(data=sub, x=x_col, y=y_col, height=figsize[0])
    g.plot_joint(sns.scatterplot, s=30, alpha=0.6)
    g.plot_marginals(sns.histplot, bins=40, kde=True)

    # Set log scales safely
    g.ax_joint.set_xscale("log")
    g.ax_joint.set_yscale("log")

    g.ax_joint.set_xlabel(f"{x_col} (log scale)")
    g.ax_joint.set_ylabel(f"{y_col} (log scale)")
    g.fig.suptitle(f"{y_col} vs {x_col} (log-log)", fontsize=14)
    g.fig.tight_layout()
    return g.fig