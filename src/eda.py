# src/eda.py
"""
Exploratory Data Analysis utilities for insurance portfolio.

Functions return pandas DataFrames or matplotlib/seaborn figure objects
so they can be used interactively in notebooks or scripts.

Author: Generated for your project
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

def default_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return a best-effort list of numeric columns used in this project."""
    candidates = [
        "TotalPremium", "TotalClaims", "CustomValueEstimate", "SumInsured",
        "CalculatedPremiumPerTerm", "Kilowatts", "cubiccapacity", "Cylinders",
        "NumberOfDoors", "CapitalOutstanding", "NumberOfVehiclesInFleet"
    ]
    return [c for c in candidates if c in df.columns]

# -------------------------
# Data Summarization
# -------------------------
def descriptive_stats(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return descriptive statistics (including variability measures) for numeric columns.
    Includes mean, std, min, 25%, median, 75%, max, IQR, coef of variation.
    """
    if numeric_cols is None:
        numeric_cols = default_numeric_columns(df)
    stats = df[numeric_cols].describe().T
    stats["IQR"] = stats["75%"] - stats["25%"]
    stats["coef_var"] = stats["std"] / stats["mean"]
    stats = stats[["count","mean","std","min","25%","50%","75%","max","IQR","coef_var"]]
    return stats

def dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return data types and a small sample of unique values for categorical inspection."""
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": df.isna().mean()
    })
    # sample some unique values for object/categorical columns
    sample_vals = {}
    for c in df.columns:
        if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]):
            sample_vals[c] = ", ".join(map(str, df[c].dropna().unique()[:5]))
        else:
            sample_vals[c] = ""
    report["sample_values"] = pd.Series(sample_vals)
    return report

# -------------------------
# Data Quality Assessment
# -------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missingness count and percentage sorted by percentage desc."""
    s = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean()
    }).sort_values("missing_pct", ascending=False)
    return s

# -------------------------
# Univariate Analysis
# -------------------------
def top_categories(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    """Return top N categories and their counts and percents."""
    counts = df[col].fillna("Unknown").value_counts().reset_index()
    counts.columns = [col, "count"]
    counts["pct"] = counts["count"] / counts["count"].sum()
    return counts.head(top_n)

# -------------------------
# Outlier Detection
# -------------------------
def detect_outliers_iqr(df: pd.DataFrame, cols: Optional[List[str]] = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Mark outliers using IQR rule.
    Returns a DataFrame with columns: col, lower, upper, n_low, n_high
    """
    if cols is None:
        cols = default_numeric_columns(df)
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        ser = df[c].dropna()
        if ser.empty:
            rows.append((c, np.nan, np.nan, 0, 0))
            continue
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_low = int((df[c] < lower).sum())
        n_high = int((df[c] > upper).sum())
        rows.append((c, lower, upper, n_low, n_high))
    out = pd.DataFrame(rows, columns=["column","lower","upper","n_low","n_high"])
    return out

# -------------------------
# Bivariate / Multivariate Analysis
# -------------------------
def correlation_matrix(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, method: str = "spearman") -> pd.DataFrame:
    """Compute correlation matrix for numeric columns. Default method: spearman (robust to non-normal)."""
    if numeric_cols is None:
        numeric_cols = default_numeric_columns(df)
    return df[numeric_cols].corr(method=method)

def monthly_aggregates(df: pd.DataFrame,
                       date_col: str = "TransactionMonth",
                       agg_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate TotalPremium and TotalClaims by month period.
    Expects TransactionMonth to be datetime or parsable.
    Returns a DataFrame indexed by Period (YYYY-MM).
    """
    if agg_cols is None:
        agg_cols = ["TotalPremium","TotalClaims"]
    df = df.copy()
    # if TransactionMonth is not datetime, try to convert
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["month"] = df[date_col].dt.to_period("M")
    monthly = df.groupby("month")[agg_cols].sum().reset_index()
    monthly["loss_ratio"] = monthly["TotalClaims"] / monthly["TotalPremium"].replace({0: np.nan})
    return monthly

def monthly_by_zip_scatter(df: pd.DataFrame, zipcode_col: str = "PostalCode") -> pd.DataFrame:
    """
    Prepare monthly changes dataset grouped by zipcode (PostalCode).
    Returns DataFrame with zipcode, month, total_premium, total_claims.
    Suitable for scatter plotting or correlation analysis.
    """
    df = df.copy()
    df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
    df["month"] = df["TransactionMonth"].dt.to_period("M")
    grouped = df.groupby([zipcode_col, "month"]).agg(
        total_premium=("TotalPremium","sum"),
        total_claims=("TotalClaims","sum"),
        n_policies=("PolicyID","nunique")
    ).reset_index()
    # compute month-to-month changes (pct change) per zipcode
    grouped = grouped.sort_values([zipcode_col,"month"])
    grouped["premium_pct_change"] = grouped.groupby(zipcode_col)["total_premium"].pct_change()
    grouped["claims_pct_change"] = grouped.groupby(zipcode_col)["total_claims"].pct_change()
    return grouped

# -------------------------
# Top/Bottom Makes & Models
# -------------------------
def top_makes_models(df: pd.DataFrame, by: str = "TotalClaims", top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return top makes and top models aggregated by 'by' column (TotalClaims or loss).
    Returns two DataFrames: top_makes, top_models
    """
    df = df.copy()
    # Make
    makes = df.groupby("Make").agg(
        total_claims=("TotalClaims","sum"),
        total_premium=("TotalPremium","sum"),
        n_policies=("PolicyID","nunique")
    ).reset_index()
    makes["loss_ratio"] = makes["total_claims"] / makes["total_premium"].replace({0: np.nan})
    makes = makes.sort_values(by if by in makes.columns else "total_claims", ascending=False)
    # Model (Make + Model)
    df["MakeModel"] = df["Make"].astype(str).str.strip() + " - " + df["Model"].astype(str).str.strip()
    models = df.groupby("MakeModel").agg(
        total_claims=("TotalClaims","sum"),
        total_premium=("TotalPremium","sum"),
        n_policies=("PolicyID","nunique")
    ).reset_index()
    models["loss_ratio"] = models["total_claims"] / models["total_premium"].replace({0: np.nan})
    models = models.sort_values(by if by in models.columns else "total_claims", ascending=False)
    return makes.head(top_n), models.head(top_n)
