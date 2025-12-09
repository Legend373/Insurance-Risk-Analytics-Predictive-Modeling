# src/models.py
"""
Modeling module for AlphaCare Insurance Solutions
- Prepares data (imputation, encoding, scaling)
- Trains regression models (severity, premium) and classification (claim probability)
- Evaluates models and computes risk-based premiums
- Provides feature importance and SHAP explanation utilities

Dependencies:
    pandas, numpy, scikit-learn, xgboost, shap, joblib
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator

# xgboost
import xgboost as xgb

# explainability
import shap

# persistence
import joblib
import os

# -------------------------
# Helpers: evaluation metrics
# -------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"rmse": float(rmse(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}

def classification_report_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Returns metrics + confusion matrix + text classification_report.
    """
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    if y_score is not None:
        try:
            res["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            res["auc"] = np.nan
    else:
        res["auc"] = np.nan

    return res


# -------------------------
# Data preparation pipeline
# -------------------------
def build_preprocessor(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build ColumnTransformer for preprocessing numeric and categorical features:
      - numeric: SimpleImputer (median) + StandardScaler
      - categorical: SimpleImputer (constant 'Unknown') + OneHotEncoder(handle_unknown='ignore')
    Returns ColumnTransformer.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    return preprocessor

# -------------------------
# Feature engineering helper
# -------------------------
def default_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common engineered features:
      - VehicleAge from RegistrationYear
      - LossRatio if available
      - ClaimCountFlag or has_claim is already computed elsewhere
    """
    df = df.copy()
    if "RegistrationYear" in df.columns:
        current_year = pd.Timestamp.now().year
        df["VehicleAge"] = df["RegistrationYear"].apply(lambda x: (current_year - int(x)) if (not pd.isna(x)) else np.nan)
    if "TotalPremium" in df.columns and "TotalClaims" in df.columns:
        df["LossRatio"] = df.apply(lambda r: np.nan if pd.isna(r["TotalPremium"]) or r["TotalPremium"] == 0 else r["TotalClaims"] / r["TotalPremium"], axis=1)
    return df

# -------------------------
# Model Trainer class
# -------------------------
class ModelTrainer:
    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str],
                 categorical: List[str],
                 numeric: List[str],
                 target_reg: str = "TotalClaims",
                 target_premium: str = "CalculatedPremiumPerTerm",
                 target_claim_flag: str = "has_claim",
                 random_state: int = 42):
        """
        df: cleaned dataframe (use Preprocessor output)
        features: list of columns to include
        categorical: list of categorical feature names
        numeric: list of numeric feature names
        """
        self.df = df.copy()
        self.features = features
        self.categorical = categorical
        self.numeric = numeric
        self.target_reg = target_reg
        self.target_premium = target_premium
        self.target_claim_flag = target_claim_flag
        self.random_state = random_state
        # placeholders
        self.preprocessor = build_preprocessor(self.df, numeric_features=self.numeric, categorical_features=self.categorical)
        self.models = {}

    # -------------------------
    # Train regression (for severity or premium)
    # -------------------------
    def train_regression(self, model_type: str = "rf", X: pd.DataFrame = None, y: pd.Series = None, test_size: float = 0.2) -> Dict:
        """
        model_type: 'linear', 'rf', 'xgb'
        Returns dict with fitted model (pipeline), X_train/X_test, y_train/y_test and metrics.
        """
        if X is None or y is None:
            X = self.df[self.features].copy()
            y = self.df[y.name] if hasattr(y, "name") else self.df[self.target_reg]
        # pipeline: preprocessor + estimator
        estimator = None
        if model_type == "linear":
            estimator = LinearRegression()
        elif model_type == "rf":
            estimator = RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        elif model_type == "xgb":
            estimator = xgb.XGBRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1, objective="reg:squarederror")
        else:
            raise ValueError("Unsupported model_type")

        pipeline = Pipeline(steps=[("preprocessor", self.preprocessor), ("estimator", estimator)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = regression_report(y_test.values, y_pred)
        # store
        key = f"reg_{model_type}"
        self.models[key] = pipeline
        return {"model_key": key, "pipeline": pipeline, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "metrics": metrics}

    # -------------------------
    # Train classifier (claim probability)
    # -------------------------
    def train_classifier(self, model_type: str = "rf", test_size: float = 0.2) -> Dict:
     """
     Train classifier to predict has_claim flag.
     Includes confusion_matrix and classification_report.
     """
     X = self.df[self.features].copy()
     y = self.df[self.target_claim_flag].astype(int)

     # Choose estimator
     if model_type == "rf":
        estimator = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
     elif model_type == "xgb":
        estimator = xgb.XGBClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss"
        )
     else:
        raise ValueError("Unsupported classifier")

     pipeline = Pipeline(steps=[
        ("preprocessor", self.preprocessor),
        ("estimator", estimator)
     ])

     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=self.random_state,
        stratify=y
     )

     pipeline.fit(X_train, y_train)
     y_pred = pipeline.predict(X_test)

     # Get score/probability if available
     y_score = None
     if hasattr(pipeline.named_steps["estimator"], "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]

     metrics = classification_report_metrics(y_test.values, y_pred, y_score)

     # Store model
     key = f"class_{model_type}"
     self.models[key] = pipeline

     # Add raw classification_report (text) and confusion matrix
     metrics["classification_report_text"] = classification_report(y_test.values, y_pred, zero_division=0)
     metrics["confusion_matrix"] = confusion_matrix(y_test.values, y_pred).tolist()

     return {
        "model_key": key,
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics
     }

    # -------------------------
    # Feature importance (for tree models)
    # -------------------------
    def feature_importance(self, model_key: str, top_n: int = 20) -> pd.DataFrame:
        """
        Compute feature importance for tree-based models.
        Returns DataFrame with feature name and importance.
        """
        pipeline = self.models.get(model_key)
        if pipeline is None:
            raise KeyError("Model not found. Train model first.")
        estimator = pipeline.named_steps["estimator"]
        # need feature names after one-hot encoding
        # get transformer feature names
        pre = pipeline.named_steps["preprocessor"]
        num_feats = pre.transformers_[0][2]
        ohe = pre.transformers_[1][1].named_steps["onehot"]
        cat_feats = pre.transformers_[1][2]
        try:
            ohe_names = list(ohe.get_feature_names_out(cat_feats))
        except Exception:
            ohe_names = []
        feat_names = list(num_feats) + ohe_names
        # importance
        if hasattr(estimator, "feature_importances_"):
            imps = estimator.feature_importances_
            df_imp = pd.DataFrame({"feature": feat_names, "importance": imps})
            df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
            return df_imp.head(top_n)
        else:
            raise ValueError("Estimator has no feature_importances_ (not a tree model)")

    # -------------------------
    # SHAP explanations
    # -------------------------
    def explain_shap(self, model_key: str, X_sample: pd.DataFrame, nsamples: int = 200):
        """
        Compute SHAP values for a sample using an appropriate explainer for model type.
        Returns explainer and shap values array.
        """
        pipeline = self.models.get(model_key)
        if pipeline is None:
            raise KeyError("Model must be trained first.")
        estimator = pipeline.named_steps["estimator"]
        pre = pipeline.named_steps["preprocessor"]

        # transform X_sample to numeric array
        X_trans = pre.transform(X_sample)
        # if tree model, use TreeExplainer on underlying estimator
        try:
            if isinstance(estimator, (RandomForestRegressor, RandomForestClassifier, xgb.XGBRegressor, xgb.XGBClassifier)):
                expl = shap.TreeExplainer(estimator)
            else:
                expl = shap.KernelExplainer(lambda x: pipeline.predict(x), shap.sample(X_trans, nsamples))
            shap_values = expl.shap_values(X_trans)
            return expl, shap_values
        except Exception as e:
            raise RuntimeError(f"SHAP failed: {e}")

    # -------------------------
    # Save / load models
    # -------------------------
    def save_model(self, model_key: str, path: str):
        pipeline = self.models.get(model_key)
        if pipeline is None:
            raise KeyError("Model not found")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(pipeline, path)
        return path

    def load_model(self, path: str, model_key: str):
        pipeline = joblib.load(path)
        self.models[model_key] = pipeline
        return pipeline

    # -------------------------
    # Utility: predict risk-based premium
    # -------------------------
    def compute_risk_based_premium(self, classifier_key: str, severity_key: str, X: pd.DataFrame, expense_loading: float = 0.05, profit_margin: float = 0.10) -> pd.Series:
        """
        premium = P(claim) * E[severity] + expense_loading * base + profit_margin * base
        Here base can be predicted premium or sum_insured. Simpler: 
           premium = P(claim) * E[severity] * (1 + expense_loading + profit_margin)
        Returns series of suggested premiums.
        """
        clf = self.models.get(classifier_key)
        sev = self.models.get(severity_key)
        if clf is None or sev is None:
            raise KeyError("Both classifier and severity models must be trained and present")

        p_claim = clf.predict_proba(X)[:, 1]
        pred_sev = sev.predict(X)
        base_risk = p_claim * pred_sev
        suggested = base_risk * (1 + expense_loading + profit_margin)
        return pd.Series(suggested, index=X.index)

# End of file
