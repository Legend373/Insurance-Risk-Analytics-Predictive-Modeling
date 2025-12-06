import numpy as np
import pandas as pd
from scipy.stats import shapiro, anderson, skew, kurtosis

class StatisticalDistributionChecker:
    """
    Standalone statistical distribution diagnostic tool.
    - Normality tests (Shapiro for n<=5000, Anderson for large n)
    - Skewness / Kurtosis
    - Poisson-likeness check (variance-to-mean ratio)
    """

    def __init__(self, max_shapiro=5000):
        self.max_shapiro = max_shapiro

    # ------------------------------
    # CORE API
    # ------------------------------
    def analyze_column(self, series: pd.Series) -> dict:
        """
        Analyze a single numeric column.
        Returns dict with statistics + interpretations.
        """
        s = series.dropna()

        result = {
            "count": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "skewness": skew(s),
            "kurtosis": kurtosis(s),
        }

        # --- Normality Test ---
        if len(s) == 0:
            result["normality_test"] = "No data"
            result["normality_p_value"] = np.nan
            result["normality_is_normal"] = False
        else:
            if len(s) <= self.max_shapiro:
                stat, p = shapiro(s)
                result["normality_test"] = "Shapiro-Wilk"
                result["normality_p_value"] = p
                result["normality_is_normal"] = p > 0.05
            else:
                ad = anderson(s)
                result["normality_test"] = "Anderson-Darling"
                result["normality_p_value"] = np.nan
                result["normality_is_normal"] = ad.statistic < ad.critical_values[2]  # 5% level

        # --- Poisson Check ---
        if result["mean"] > 0:
            vmr = (result["std"] ** 2) / result["mean"]
            result["poisson_vmr"] = vmr
            result["poisson_like"] = abs(vmr - 1) < 0.2
        else:
            result["poisson_vmr"] = np.nan
            result["poisson_like"] = False

        # --- Interpretation ---
        result["interpretation"] = self._interpret(result)

        return result

    # ------------------------------
    # MULTI-COLUMN API
    # ------------------------------
    def analyze_dataframe(self, df: pd.DataFrame, numeric_only=True) -> pd.DataFrame:
        """
        Analyze all numeric columns and return DataFrame summary.
        """
        if numeric_only:
            df = df.select_dtypes(include=[np.number])

        results = {}
        for col in df.columns:
            results[col] = self.analyze_column(df[col])

        return pd.DataFrame(results).T

    # ------------------------------
    # INTERNAL INTERPRETATION ENGINE
    # ------------------------------
    def _interpret(self, r: dict) -> str:
        msgs = []

        # Normality
        if r["normality_is_normal"]:
            msgs.append("Distribution appears normal.")
        else:
            msgs.append("Not normally distributed → use Spearman or non-parametric tests.")

        # Skewness
        if abs(r["skewness"]) > 1:
            msgs.append("Highly skewed → consider log or Box-Cox transform.")
        elif abs(r["skewness"]) > 0.5:
            msgs.append("Moderately skewed.")

        # Kurtosis
        if r["kurtosis"] > 3:
            msgs.append("Heavy-tailed distribution.")

        # Poisson
        if r["poisson_like"]:
            msgs.append("Variance≈mean → Poisson-like behavior detected.")

        return " ".join(msgs) if msgs else "No notable patterns detected."
