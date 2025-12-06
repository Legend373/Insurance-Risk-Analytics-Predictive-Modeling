# src/preprocessor/preprocessor_v2.py
"""
Preprocessor v2 - robust and safe preprocessing for the insurance portfolio dataset.

Key fixes over previous version:
- read_raw: auto-detect header, correct indentation
- normalize_strings: robust trimming and blank -> NaN conversion
- _clean_numeric_token: much safer numeric parsing (doesn't coerce everything to NaN)
- categorical handling uses pandas StringDtype to avoid literal "nan" strings
- outlier flags created reliably (kept separate columns, not overwritten)
- missing_summary returns a DataFrame
- logging (print) at main steps for interactive notebooks

Usage:
    from src.preprocessor.preprocessor_v2 import PreprocessorV2
    pre = PreprocessorV2()
    df_clean = pre.run(infile="data/raw/insurance_data.txt", save_to="data/clean/clean_portfolio_v2.csv")
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import numpy as np

# Column names (expected) — used if file has no header
COLUMN_NAMES = [
    "UnderwrittenCoverID","PolicyID","TransactionMonth","IsVATRegistered",
    "Citizenship","LegalType","Title","Language","Bank","AccountType",
    "MaritalStatus","Gender","Country","Province","PostalCode","MainCrestaZone",
    "SubCrestaZone","ItemType","mmcode","VehicleType","RegistrationYear",
    "Make","Model","Cylinders","cubiccapacity","kilowatts","bodytype",
    "NumberOfDoors","VehicleIntroDate","CustomValueEstimate","AlarmImmobiliser",
    "TrackingDevice","CapitalOutstanding","NewVehicle","WrittenOff","Rebuilt",
    "Converted","CrossBorder","NumberOfVehiclesInFleet","SumInsured",
    "TermFrequency","CalculatedPremiumPerTerm","ExcessSelected","CoverCategory",
    "CoverType","CoverGroup","Section","Product","StatutoryClass",
    "StatutoryRiskType","TotalPremium","TotalClaims"
]


class PreprocessorV2:
    def __init__(self, sep: str = "|", expected_cols: Optional[List[str]] = None):
        self.sep = sep
        self.expected_cols = expected_cols or COLUMN_NAMES
        # tokens we explicitly treat as zero (but handled conservatively)
        self._explicit_zero_tokens = {".000000000000", ".000000000", "0.000000000", "0", "0.0"}
        # characters to strip from numeric tokens
        self._numeric_strip_chars = [",", "$", "R", " "]

    # -------------------------
    # Reading
    # -------------------------
    def read_raw(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Read a pipe-delimited file. Auto-detects whether file contains header row
        by comparing first-row column labels to expected columns.
        Always returns a DataFrame with string/object dtypes initially.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        # Read only first row to inspect header
        try:
            first = pd.read_csv(p, sep=self.sep, nrows=0)
            has_header = len(set(first.columns).intersection(set(self.expected_cols))) > 3
        except Exception:
            has_header = False

        if has_header:
            # read with header
            df = pd.read_csv(p, sep=self.sep, header=0, dtype=str, low_memory=False)
        else:
            # read without header, apply expected column names
            df = pd.read_csv(p, sep=self.sep, header=None, names=self.expected_cols, dtype=str, low_memory=False)

        # Ensure we have all expected columns (add missing as NaN)
        for c in self.expected_cols:
            if c not in df.columns:
                df[c] = pd.NA

        print(f"[PreprocessorV2] Loaded {len(df):,} rows; columns={len(df.columns)}; header_detected={has_header}")
        return df

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _to_bool_series(s: pd.Series) -> pd.Series:
        s2 = s.astype("string").str.strip().str.lower().replace("", pd.NA)
        return s2.map({"true": True, "false": False, "yes": True, "no": False}).astype("boolean")

    def _clean_numeric_token(self, v) -> Optional[float]:
        """
        Safely convert an input token to float.
        - Returns float if convertible
        - Treats known explicit zero tokens as 0.0
        - Returns np.nan for values that cannot be reasonably parsed
        """
        if pd.isna(v):
            return np.nan
        v_str = str(v).strip()
        if v_str == "":
            return np.nan

        # explicit zero tokens
        if v_str in self._explicit_zero_tokens:
            return 0.0

        # remove thousands separators and currency symbols
        for ch in self._numeric_strip_chars:
            v_str = v_str.replace(ch, "")

        # handle cases like '.000000000000' -> treat as zero if all zeros after dot
        if v_str.startswith(".") and set(v_str[1:]) <= set("0"):
            return 0.0

        # Now check if it's a legitimate numeric string (allow leading +/-, decimal)
        # Accept if at most one dot and remaining chars are digits or leading +/-.
        v_test = v_str
        if v_test.startswith(("+", "-")):
            v_test = v_test[1:]
        # allow decimal numbers and digits
        if v_test.replace(".", "", 1).isdigit():
            try:
                return float(v_str)
            except Exception:
                return np.nan

        # sometimes numbers come with parentheses for negatives e.g. (1234)
        if v_str.startswith("(") and v_str.endswith(")"):
            inner = v_str[1:-1].replace(",", "")
            if inner.replace(".", "", 1).isdigit():
                try:
                    return -float(inner)
                except:
                    return np.nan

        # otherwise give up -> NaN
        return np.nan

    # -------------------------
    # Step 1: Normalize strings & blanks
    # -------------------------
    def normalize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Replace pure whitespace-only strings with NaN across all object/string columns
        for c in df.columns:
            # convert to pandas string dtype to handle missing nicely
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                # keep as StringDtype where possible
                df[c] = df[c].astype("string").str.strip()
                # empty strings -> <NA>
                df[c] = df[c].replace("", pd.NA)
        return df

    # -------------------------
    # Step 2: Parse dates
    # -------------------------
    # -------------------------
# Step 2: Parse dates
# -------------------------
    def parse_dates(self, df: pd.DataFrame, date_cols: Optional[List[str]] = None) -> pd.DataFrame:
     df = df.copy()
     if date_cols is None:
        date_cols = ["TransactionMonth", "VehicleIntroDate"]
     for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")  # removed deprecated infer_datetime_format
    # Derived period
     if "TransactionMonth" in df.columns:
        df["TransactionPeriod"] = pd.to_datetime(df["TransactionMonth"], errors="coerce").dt.to_period("M")
     return df

# -------------------------
# Step 3: Convert types
# -------------------------
    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
      df = df.copy()

    # Booleans (map common tokens)
      bool_cols = [
        "IsVATRegistered", "AlarmImmobiliser", "TrackingDevice",
        "NewVehicle", "WrittenOff", "Rebuilt", "Converted", "CrossBorder"
      ]
      for c in bool_cols:
        if c in df.columns:
            df[c] = self._to_bool_series(df[c])

    # Numeric conversion using safe parser
      numeric_candidates = [
        "CustomValueEstimate", "CapitalOutstanding", "NumberOfVehiclesInFleet",
        "SumInsured", "CalculatedPremiumPerTerm", "TotalPremium", "TotalClaims",
        "Kilowatts", "cubiccapacity", "Cylinders", "NumberOfDoors"
      ]
      for c in numeric_candidates:
        if c in df.columns:
            df[c] = df[c].apply(self._clean_numeric_token)

    # RegistrationYear: integer-like
      if "RegistrationYear" in df.columns:
        df["RegistrationYear"] = pd.to_numeric(df["RegistrationYear"], errors="coerce").astype("Int64")

    # PostalCode keep as string
      if "PostalCode" in df.columns:
        df["PostalCode"] = df["PostalCode"].astype("string").str.strip().replace("", pd.NA)

    # Categorical normalization using string dtype
      cat_cols = ["Make", "Model", "Province", "VehicleType", "Gender", "Country", "CoverCategory", "CoverType", "Product"]
      for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
            # normalize obvious placeholders to NA
            mask = df[c].str.lower().isin(["nan", "none", "none.", "not specified", "<na>"])
            df.loc[mask, c] = pd.NA

      return df

    # -------------------------
    # Step 4: Missing value strategy
    # -------------------------
    def impute_missing(self, df: pd.DataFrame, numeric_fill: Optional[float] = None) -> pd.DataFrame:
        df = df.copy()
        # numeric_fill: if provided, fill numeric columns with this value
        if numeric_fill is not None:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(numeric_fill)

        # categorical (string) fill with "Unknown" for convenience in grouping (but keep NA if you prefer)
        str_cols = df.select_dtypes(include=["string"]).columns
        df[str_cols] = df[str_cols].fillna("Unknown")

        # boolean fill with False (pandas boolean dtype)
        bool_cols = df.select_dtypes(include=["boolean"]).columns
        df[bool_cols] = df[bool_cols].fillna(False)

        return df

    # -------------------------
    # Step 5: Deduplicate & validate
    # -------------------------
    def deduplicate_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        key_cols = [c for c in ["PolicyID", "TransactionMonth", "UnderwrittenCoverID"] if c in df.columns]
        if key_cols:
            before = len(df)
            df = df.drop_duplicates(subset=key_cols)
            after = len(df)
            if before != after:
                print(f"[PreprocessorV2] Dropped {before-after} duplicates using keys {key_cols}")
        else:
            df = df.drop_duplicates()
        # Ensure TotalPremium & TotalClaims numeric (they should already be)
        if "TotalPremium" in df.columns:
            df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")
        if "TotalClaims" in df.columns:
            df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")
        return df

    # -------------------------
    # Step 6: Outlier detection & flags (IQR) - creates flags, does NOT overwrite original unless clip=True
    # -------------------------
    def outlier_flags_iqr(self, df: pd.DataFrame, cols: Optional[List[str]] = None, factor: float = 1.5, clip: bool = False) -> pd.DataFrame:
        """
        Add <col>_outlier_flag columns with values: 'low', 'high', or NaN.
        If clip=True will clip values to the IQR bounds (not default).
        """
        df = df.copy()
        if cols is None:
            cols = ["TotalPremium", "TotalClaims", "CustomValueEstimate", "SumInsured"]

        for c in cols:
            if c not in df.columns:
                continue
            ser = df[c].dropna()
            if ser.empty:
                df[f"{c}_outlier_flag"] = pd.NA
                continue
            q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            flag_col = f"{c}_outlier_flag"
            df[flag_col] = pd.NA
            df.loc[df[c] < lower, flag_col] = "low"
            df.loc[df[c] > upper, flag_col] = "high"
            if clip:
                df[c] = df[c].clip(lower=lower, upper=upper)
        return df

    # -------------------------
    # Step 7: Feature engineering
    # -------------------------
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # LossRatio
        if {"TotalClaims", "TotalPremium"}.issubset(df.columns):
            def lr_func(row):
                tp = row["TotalPremium"]
                tc = row["TotalClaims"]
                if pd.isna(tp) or tp == 0:
                    return np.nan
                return tc / tp
            df["LossRatio"] = df.apply(lr_func, axis=1)

        # Transaction year/month/period (if TransactionMonth present)
        if "TransactionMonth" in df.columns:
            df["Trans_Year"] = pd.to_datetime(df["TransactionMonth"], errors="coerce").dt.year
            df["Trans_Month"] = pd.to_datetime(df["TransactionMonth"], errors="coerce").dt.month
            df["Trans_YearMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce").dt.to_period("M")

        # Vehicle age (if RegistrationYear present)
        if "RegistrationYear" in df.columns:
            current_year = pd.Timestamp.now().year
            # RegistrationYear is Int64 (nullable); coerce to numeric then compute
            df["VehicleAge"] = df["RegistrationYear"].apply(lambda x: (current_year - int(x)) if not pd.isna(x) else pd.NA)

        return df

    # -------------------------
    # Master run pipeline
    # -------------------------
    def run(self,
            df: Optional[pd.DataFrame] = None,
            infile: Optional[Union[str, Path]] = None,
            save_to: Optional[Union[str, Path]] = "data/clean/clean_portfolio_v2.csv",
            impute_numeric_with: Optional[float] = None,
            create_outlier_flags: bool = True,
            clip_outliers: bool = False
        ) -> pd.DataFrame:
        """
        End-to-end preprocessing. Provide either df or infile.
        - impute_numeric_with: if not None, numeric NaNs will be filled with this value.
        - create_outlier_flags: create <col>_outlier_flag columns
        - clip_outliers: if True, numeric values will be clipped to IQR bounds (use with caution)
        Saves to CSV if save_to provided.
        """
        if df is None and infile is None:
            raise ValueError("Either df or infile must be provided.")
        if df is None:
            df = self.read_raw(infile)

        print("[PreprocessorV2] Step 1 - normalize strings")
        df = self.normalize_strings(df)

        print("[PreprocessorV2] Step 2 - parse dates")
        df = self.parse_dates(df)

        print("[PreprocessorV2] Step 3 - convert types")
        df = self.convert_types(df)

        print("[PreprocessorV2] Step 4 - impute missing (categorical -> Unknown, boolean -> False)")
        df = self.impute_missing(df, numeric_fill=impute_numeric_with)

        print("[PreprocessorV2] Step 5 - deduplicate & validate")
        df = self.deduplicate_and_validate(df)

        if create_outlier_flags:
            print("[PreprocessorV2] Step 6 - create outlier flags (IQR)")
            df = self.outlier_flags_iqr(df, clip=clip_outliers)

        print("[PreprocessorV2] Step 7 - feature engineering")
        df = self.add_features(df)

        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"[PreprocessorV2] Cleaned data saved to {save_path.resolve()}")

        return df

    # -------------------------
    # Utility: missing summary
    # -------------------------
    @staticmethod
    def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
        s = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean()
        }).sort_values("missing_pct", ascending=False)
        return s
