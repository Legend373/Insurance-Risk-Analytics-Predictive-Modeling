"""
src/hypothesis_tests.py

Hypothesis testing utilities for insurance analytics (Task 3).

Provides:
- KPI computation (has_claim, claim_severity, margin)
- Group tests: frequency (chi-square + Cramer's V), severity/margin (ANOVA / Kruskal-Wallis)
- Pairwise t-tests and effect sizes (Cohen's d)
- Regression helpers (logit for frequency, OLS for numeric outcomes)
- Multiple-testing correction helper
- Simple reporting helpers

Dependencies:
    pandas, numpy, scipy, statsmodels
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# -------------------------
# KPI helpers
# -------------------------
def compute_kpis(df: pd.DataFrame,
                 claim_col: str = "TotalClaims",
                 premium_col: str = "TotalPremium") -> pd.DataFrame:
    """
    Add KPI columns:
      - has_claim: bool, True if TotalClaims > 0
      - claim_severity: value of TotalClaims when has_claim else NaN
      - margin: TotalPremium - TotalClaims
    Returns a copy of df with new columns.
    """
    df = df.copy()
    df["has_claim"] = df[claim_col].fillna(0) > 0
    df["claim_severity"] = df[claim_col].where(df["has_claim"], other=np.nan)
    df["margin"] = df[premium_col].fillna(0) - df[claim_col].fillna(0)
    return df

# -------------------------
# Effect sizes
# -------------------------
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples (use pooled std)."""
    x = np.asarray(x[~np.isnan(x)])
    y = np.asarray(y[~np.isnan(y)])
    nx, ny = len(x), len(y)
    if nx < 1 or ny < 1:
        return np.nan
    sx = np.std(x, ddof=1) if nx > 1 else 0.0
    sy = np.std(y, ddof=1) if ny > 1 else 0.0
    pooled = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / max(nx + ny - 2, 1))
    if pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled

def cramers_v(table: pd.DataFrame) -> float:
    """Compute Cramer's V from contingency table (pandas crosstab)."""
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    k = min(table.shape) - 1
    if k == 0:
        return 0.0
    return np.sqrt(chi2 / (n * k))

# -------------------------
# Frequency tests (categorical)
# -------------------------
def test_frequency_by_group(df: pd.DataFrame, group_col: str, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Chi-square test for independence between group_col and has_claim.
    Returns dictionary with table, chi2, p_value, dof, expected, cramers_v, reject_H0.
    """
    if group_col not in df.columns:
        raise KeyError(f"{group_col} not in DataFrame")
    table = pd.crosstab(df[group_col].fillna("Unknown"), df["has_claim"])
    if table.to_numpy().sum() == 0:
        return {"error": "empty contingency table"}
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    v = cramers_v(table)
    return {
        "table": table,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": expected,
        "cramers_v": float(v),
        "reject_H0": p < alpha
    }

# -------------------------
# Numeric group tests: severity / margin
# -------------------------
def test_numeric_by_group(df: pd.DataFrame,
                          value_col: str,
                          group_col: str,
                          alpha: float = 0.05,
                          nonparametric_if_needed: bool = True) -> Dict[str, Any]:
    """
    Test whether the distribution of value_col differs across groups defined by group_col.
    For severity use value_col='claim_severity' (drop NaNs), for margin use value_col='margin' (use all).
    Returns test type ('anova' or 'kruskal'), statistic, p_value, levene_p, reject_H0.
    """
    if group_col not in df.columns:
        raise KeyError(f"{group_col} not in DataFrame")
    if value_col not in df.columns:
        raise KeyError(f"{value_col} not in DataFrame")

    # Prepare groups
    grouped = df.groupby(group_col)[value_col].apply(lambda s: s.dropna().values)
    groups = [g for g in grouped if len(g) > 0]
    if len(groups) < 2:
        return {"error": "Not enough groups with data"}

    # Levene test for homogeneity of variance
    try:
        levene_stat, levene_p = stats.levene(*groups)
    except Exception:
        levene_stat, levene_p = np.nan, np.nan

    # ANOVA
    try:
        anova_stat, anova_p = stats.f_oneway(*groups)
    except Exception:
        anova_stat, anova_p = np.nan, np.nan

    if nonparametric_if_needed and (not np.isnan(levene_p) and levene_p < 0.05):
        # variance heterogeneity -> Kruskal-Wallis
        try:
            kw_stat, kw_p = stats.kruskal(*groups)
        except Exception:
            kw_stat, kw_p = np.nan, np.nan
        return {
            "test": "kruskal",
            "statistic": float(kw_stat) if not np.isnan(kw_stat) else np.nan,
            "p_value": float(kw_p) if not np.isnan(kw_p) else np.nan,
            "levene_p": float(levene_p) if not np.isnan(levene_p) else np.nan,
            "reject_H0": (not np.isnan(kw_p)) and (kw_p < alpha)
        }
    else:
        return {
            "test": "anova",
            "statistic": float(anova_stat) if not np.isnan(anova_stat) else np.nan,
            "p_value": float(anova_p) if not np.isnan(anova_p) else np.nan,
            "levene_p": float(levene_p) if not np.isnan(levene_p) else np.nan,
            "reject_H0": (not np.isnan(anova_p)) and (anova_p < alpha)
        }

# -------------------------
# Pairwise comparisons with multiple-testing correction
# -------------------------
def pairwise_comparisons(df: pd.DataFrame,
                         value_col: str,
                         group_col: str,
                         correction_method: str = "fdr_bh",
                         alpha: float = 0.05) -> pd.DataFrame:
    """
    Conduct pairwise t-tests between all group pairs for value_col. Returns DataFrame with:
    group1, group2, t_stat, p_value, p_adj (corrected), cohen_d, reject
    Note: if normality assumptions fail, consider replacing with mannwhitneyu outside this function.
    """
    groups = df.groupby(group_col)[value_col].apply(lambda s: s.dropna().values).to_dict()
    keys = list(groups.keys())
    rows = []
    pvals = []
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = groups[keys[i]]
            b = groups[keys[j]]
            if len(a) < 2 or len(b) < 2:
                t_stat = np.nan; p = np.nan; d = np.nan
            else:
                res = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
                t_stat = res.statistic; p = res.pvalue
                d = cohens_d(a, b)
            rows.append({"group1": keys[i], "group2": keys[j], "t_stat": t_stat, "p_value": p, "cohen_d": d})
            pvals.append(p if p is not None else np.nan)
            pairs.append((keys[i], keys[j]))
    # multiple testing correction
    pvals_clean = [0.0 if np.isnan(x) else x for x in pvals]
    reject, p_adj, _, _ = multipletests(pvals_clean, alpha=alpha, method=correction_method)
    for k, row in enumerate(rows):
        row["p_adj"] = float(p_adj[k]) if not np.isnan(p_adj[k]) else np.nan
        row["reject_H0"] = bool(reject[k])
    return pd.DataFrame(rows)

# -------------------------
# Regression helpers (adjusted effects)
# -------------------------
def logistic_regression(df: pd.DataFrame, formula: str, dependent_is_bool: bool = True) -> Any:
    """
    Fit a logistic regression using formula (statsmodels).
    If dependent_is_bool, convert True/False to 1/0.
    Returns fitted model.
    Example formula: 'has_claim ~ C(Province) + C(VehicleType) + C(Gender) + VehicleAge'
    """
    df = df.copy()
    dep = formula.split("~")[0].strip()
    if dependent_is_bool:
        df[dep] = df[dep].astype(int)
    model = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
    return model

def ols_regression(df: pd.DataFrame, formula: str) -> Any:
    """
    Fit OLS for numeric outcomes (margin or claim_severity).
    Returns fitted model.
    """
    model = smf.ols(formula, data=df).fit()
    return model

# -------------------------
# Multiple testing helper
# -------------------------
def adjust_pvalues(pvalues: List[float], method: str = "fdr_bh", alpha: float = 0.05) -> Dict[str, Any]:
    """
    Adjust a list of p-values using statsmodels multipletests.
    Returns dict: reject, p_adj, method.
    """
    pv = [0.0 if np.isnan(x) else x for x in pvalues]
    reject, p_adj, _, _ = multipletests(pv, alpha=alpha, method=method)
    return {"reject": reject, "p_adj": p_adj, "method": method}

# -------------------------
# Runner that executes the main hypotheses
# -------------------------
def run_core_hypotheses(df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Run the main hypothesis tests:
      - risk by Province (frequency & severity)
      - risk by PostalCode (frequency & severity) [caveat: high-cardinality]
      - margin by PostalCode
      - risk by Gender
    Returns dictionary of results ready for reporting.
    """
    df = compute_kpis(df)
    results = {}
    # Province: frequency & severity & margin
    results["province_freq"] = test_frequency_by_group(df, "Province", alpha=alpha)
    results["province_severity"] = test_numeric_by_group(df, "claim_severity", "Province", alpha=alpha)
    results["province_margin"] = test_numeric_by_group(df, "margin", "Province", alpha=alpha)

    # PostalCode: frequency & severity & margin (high-cardinality caution)
    if "PostalCode" in df.columns:
        results["zipcode_freq"] = test_frequency_by_group(df, "PostalCode", alpha=alpha)
        results["zipcode_severity"] = test_numeric_by_group(df, "claim_severity", "PostalCode", alpha=alpha)
        results["zipcode_margin"] = test_numeric_by_group(df, "margin", "PostalCode", alpha=alpha)
    else:
        results["zipcode_freq"] = {"error": "PostalCode not in df"}

    # Gender
    if "Gender" in df.columns:
        results["gender_freq"] = test_frequency_by_group(df, "Gender", alpha=alpha)
        results["gender_severity"] = test_numeric_by_group(df, "claim_severity", "Gender", alpha=alpha)
        results["gender_margin"] = test_numeric_by_group(df, "margin", "Gender", alpha=alpha)
    else:
        results["gender_freq"] = {"error": "Gender not in df"}

    return results

# -------------------------
# Small reporting helper
# -------------------------
def summarize_test_result(result: Dict[str, Any]) -> str:
    """
    Turn result dict into a human-readable one-line summary for reporting.
    """
    if "error" in result:
        return f"ERROR: {result['error']}"
    if "p_value" in result:
        p = result["p_value"]
        reject = result.get("reject_H0", False)
        return f"p={p:.4g} -> {'REJECT H0' if reject else 'FAIL TO REJECT H0'}"
    if "statistic" in result and "p_value" in result:
        p = result["p_value"]
        reject = result.get("reject_H0", False)
        return f"stat={result['statistic']:.4g}, p={p:.4g} -> {'REJECT H0' if reject else 'FAIL TO REJECT H0'}"
    return "No p-value found in result"

# End of module
