"""
data_loader.py
==============
Handles CSV upload, column validation, preprocessing, and feature engineering.
Returns clean DataFrames and a fitted ColumnTransformer ready for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")


# ─── Required columns (minimum viable dataset) ───────────────────────────────
REQUIRED_COLS = [
    "Age", "Attrition", "Department", "JobSatisfaction",
    "MonthlyIncome", "OverTime", "YearsAtCompany", "WorkLifeBalance"
]

# Columns we *use* if present (optional but enrich the model)
OPTIONAL_COLS = [
    "BusinessTravel", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "Gender", "JobInvolvement", "JobLevel", "JobRole", "MaritalStatus",
    "NumCompaniesWorked", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
]


def load_and_validate(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a CSV (Streamlit UploadedFile or file path), validate required columns.
    Returns (dataframe, list_of_warnings).
    Raises ValueError if required columns are missing.
    """
    warnings_list = []

    # ── Load ─────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")

    # ── Strip whitespace from column names ───────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Check required columns ───────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Your file has: {list(df.columns)}"
        )

    # ── Check Attrition column values ────────────────────────────────────────
    attrition_vals = df["Attrition"].dropna().unique()
    valid_attrition = {"Yes", "No", 1, 0, "1", "0", True, False}
    if not any(v in valid_attrition for v in attrition_vals):
        raise ValueError(
            f"'Attrition' column must contain Yes/No or 1/0. Found: {attrition_vals}"
        )

    # ── Warn about optional missing columns ──────────────────────────────────
    missing_optional = [c for c in OPTIONAL_COLS if c not in df.columns]
    if missing_optional:
        warnings_list.append(
            f"Optional columns not found (model will still work): {missing_optional}"
        )

    # ── Warn about nulls ─────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        warnings_list.append(
            f"Null values detected and will be imputed: {null_cols.to_dict()}"
        )

    return df, warnings_list


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Clean, encode, and engineer features.
    Returns:
      - X: feature DataFrame (all numeric)
      - y: binary attrition Series (0/1)
      - numeric_cols: list of numeric feature names
      - categorical_cols: list of categorical feature names
    """
    df = df.copy()

    # ── Encode target ─────────────────────────────────────────────────────────
    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    df["Attrition"] = df["Attrition"].astype(int)
    y = df["Attrition"]
    df = df.drop(columns=["Attrition"])

    # ── Drop low-information columns if present ───────────────────────────────
    drop_always = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df = df.drop(columns=[c for c in drop_always if c in df.columns])

    # ── Feature engineering ──────────────────────────────────────────────────
    if "MonthlyIncome" in df.columns and "Age" in df.columns:
        df["IncomePerYear"] = df["MonthlyIncome"] / (df["Age"].replace(0, 1))

    if "YearsAtCompany" in df.columns and "TotalWorkingYears" in df.columns:
        df["CompanyLoyaltyRatio"] = (
            df["YearsAtCompany"] / (df["TotalWorkingYears"].replace(0, 1))
        ).clip(0, 1)

    if "YearsAtCompany" in df.columns and "YearsSinceLastPromotion" in df.columns:
        df["PromotionLag"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]

    if "JobSatisfaction" in df.columns and "WorkLifeBalance" in df.columns:
        df["WellbeingScore"] = (df["JobSatisfaction"] + df["WorkLifeBalance"]) / 2

    # ── Encode OverTime ───────────────────────────────────────────────────────
    if "OverTime" in df.columns and df["OverTime"].dtype == object:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # ── Identify column types ─────────────────────────────────────────────────
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Impute ────────────────────────────────────────────────────────────────
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    return df, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer:
    - Numeric: StandardScaler
    - Categorical: OneHotEncoder (handle_unknown='ignore')
    """
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="drop")

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: list, categorical_cols: list) -> list:
    """Extract feature names after fitting the preprocessor."""
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    return numeric_cols + cat_feature_names


def get_summary_stats(df: pd.DataFrame, y: pd.Series) -> dict:
    """Return a dict of summary statistics for the Overview tab."""
    return {
        "total_employees": len(df),
        "attrition_count": int(y.sum()),
        "attrition_rate": round(float(y.mean()) * 100, 1),
        "avg_age": round(float(df["Age"].mean()), 1) if "Age" in df.columns else None,
        "avg_tenure": round(float(df["YearsAtCompany"].mean()), 1) if "YearsAtCompany" in df.columns else None,
        "avg_satisfaction": round(float(df["JobSatisfaction"].mean()), 1) if "JobSatisfaction" in df.columns else None,
        "overtime_pct": round(float((df["OverTime"] == 1).mean()) * 100, 1) if "OverTime" in df.columns else None,
    }
