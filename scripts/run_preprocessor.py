# scripts/run_preprocessor.py
from src.preprocessor.preprocessor import Preprocessor
from pathlib import Path

if __name__ == "__main__":
    raw = Path("data/raw/insurance_data.txt")
    out = Path("data/clean/clean_portfolio.csv")

    pre = Preprocessor()
    df_clean = pre.run(infile=raw, save_to=out, impute_numeric_with=None, cap_outliers=True)

    print("Rows:", len(df_clean))
    print(df_clean.columns.tolist()[:20])
    print(pre.missing_summary(df_clean).head(15))
