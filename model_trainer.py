"""
model_trainer.py
================
Trains two models:
  1. Attrition Classifier  → XGBoost with RandomForest fallback
  2. Wellness Clusterer    → KMeans + rule-based recommendation engine

Both are wrapped in sklearn Pipelines so they can be pickled and reloaded.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import pickle, os, warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ─── Wellness recommendation rules (cluster → text) ──────────────────────────
WELLNESS_RULES = {
    # High risk, low satisfaction
    "high_risk_low_sat": {
        "label": "Burnout Risk",
        "color": "#E24B4A",
        "recommendations": [
            "Schedule a 1:1 with manager to discuss workload",
            "Review overtime policy — consider mandatory comp days",
            "Offer mental health days or EAP counselling",
            "Create a structured career growth plan within 30 days",
        ]
    },
    # High risk, high overtime
    "high_risk_overtime": {
        "label": "Overload Alert",
        "color": "#EF9F27",
        "recommendations": [
            "Audit task allocation and redistribute workload",
            "Offer flexible / remote work options",
            "Introduce a 'no-meeting Friday' policy",
            "Review compensation against market benchmarks",
        ]
    },
    # Low satisfaction, low income
    "low_sat_low_income": {
        "label": "Disengaged & Underpaid",
        "color": "#D4537E",
        "recommendations": [
            "Conduct pay equity review for this cohort",
            "Implement recognition and rewards programme",
            "Provide upskilling budget (courses, certifications)",
            "Offer performance-linked bonus structure",
        ]
    },
    # Low promotion rate
    "stagnant_career": {
        "label": "Career Stagnation",
        "color": "#7F77DD",
        "recommendations": [
            "Create transparent promotion criteria",
            "Assign internal mentors from senior staff",
            "Offer lateral mobility or job rotation",
            "Set 6-month milestone reviews with clear targets",
        ]
    },
    # Generally healthy
    "healthy": {
        "label": "Healthy & Engaged",
        "color": "#1D9E75",
        "recommendations": [
            "Maintain current engagement programmes",
            "Celebrate tenure milestones publicly",
            "Involve in mentorship roles for newer employees",
            "Offer stretch assignments to maintain challenge",
        ]
    }
}


# ─── Attrition Model ──────────────────────────────────────────────────────────

def train_attrition_model(
    X_transformed: np.ndarray,
    y: pd.Series,
    feature_names: list,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Train the attrition classification model.
    Returns a results dict with model, metrics, and feature importances.
    """                                                                              
    
    
    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # ── Choose model ─────────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        base_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0
        )
    else:
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state
        )

    # ── Calibrate probabilities ───────────────────────────────────────────────
    model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # ── Cross-val AUC ─────────────────────────────────────────────────────────
    cv_scores = cross_val_score(base_model, X_transformed, y, cv=5, scoring="roc_auc")

    # ── Feature importances ───────────────────────────────────────────────────
    if hasattr(base_model, "feature_importances_"):
        importances = base_model.feature_importances_
    elif hasattr(model, "estimators_"):
        # Extract from calibrated classifier
        try:
            inner = model.calibrated_classifiers_[0].estimator
            importances = inner.feature_importances_
        except Exception:
            importances = np.zeros(len(feature_names))
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    return {
        "model": model,
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_prec, 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": importance_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob_test": y_prob,
    }


# ─── SHAP Explainer ───────────────────────────────────────────────────────────

def compute_shap_values(model, X_transformed: np.ndarray, feature_names: list, max_samples: int = 200):
    """
    Compute SHAP values for a sample of rows.
    Returns (shap_values array, sample_X array) or (None, None) if shap unavailable.
    """
    try:
        import shap
        # Use a background sample for TreeExplainer or KernelExplainer
        sample = X_transformed[:min(max_samples, len(X_transformed))]

        # Try TreeExplainer first (fast), fall back to KernelExplainer
        try:
            inner = model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(inner)
            shap_vals = explainer.shap_values(sample)
            # For binary classifiers, take class 1
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        except Exception:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(sample, 50))
            shap_vals = explainer.shap_values(sample)[:, :, 1]

        return shap_vals, sample, feature_names
    except ImportError:
        return None, None, None


# ─── Wellness Clustering Model ─────────────────────────────────────────────

def train_wellness_model(df: pd.DataFrame, n_clusters: int = 5) -> dict:
    """
    Cluster employees using KMeans on wellness-relevant numeric features.
    Maps each cluster to a wellness profile using rule-based logic.
    """
    # ── Select wellness features ────────────────────────────────────────────
    wellness_features = [
        "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "JobInvolvement", "MonthlyIncome",
        "YearsAtCompany", "YearsSinceLastPromotion", "OverTime"
    ]                                                                                  
    available = [f for f in wellness_features if f in df.columns]
    wellness_df = df[available].copy()

    # Encode OverTime if still string
    if "OverTime" in wellness_df.columns and wellness_df["OverTime"].dtype == object:
        wellness_df["OverTime"] = wellness_df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    # Impute
    for col in wellness_df.columns:
        wellness_df[col] = wellness_df[col].fillna(wellness_df[col].median())

    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_wellness = scaler.fit_transform(wellness_df)

    # Cluster
    n_clusters = min(n_clusters, len(wellness_df) - 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_wellness)

    # ── Map clusters to wellness profiles ─────────────────────────────────────
    profile_map = _assign_wellness_profiles(
        wellness_df, cluster_labels, available
    )

    return {
        "model": kmeans,
        "scaler": scaler,
        "cluster_labels": cluster_labels,
        "features_used": available,
        "profile_map": profile_map,
        "n_clusters": n_clusters,
    }


def _assign_wellness_profiles(df: pd.DataFrame, labels: np.ndarray, features: list) -> dict:
    """
    For each cluster, compute mean feature values and assign a wellness profile
    using rule-based thresholds.
    """
    df = df.copy()
    df["_cluster"] = labels
    cluster_means = df.groupby("_cluster").mean()

    profile_map = {}
    for cluster_id, row in cluster_means.iterrows():
        sat = row.get("JobSatisfaction", 2.5)
        wlb = row.get("WorkLifeBalance", 2.5)
        ot = row.get("OverTime", 0.3)
        income = row.get("MonthlyIncome", 5000)
        promo_lag = row.get("YearsSinceLastPromotion", 2)
        tenure = row.get("YearsAtCompany", 3)

        # Rule logic — order matters (most severe first)
        if sat < 2.0 and ot > 0.5:
            key = "high_risk_overtime"
        elif sat < 2.0 and wlb < 2.0:
            key = "high_risk_low_sat"
        elif sat < 2.5 and income < 4000:
            key = "low_sat_low_income"
        elif promo_lag > 4 and tenure > 3:
            key = "stagnant_career"
        else:
            key = "healthy"

        profile_map[int(cluster_id)] = {
            "profile_key": key,
            "cluster_size": int((labels == cluster_id).sum()),
            "mean_satisfaction": round(float(sat), 2),
            "mean_wlb": round(float(wlb), 2),
            "mean_overtime": round(float(ot), 2),
            "mean_income": round(float(income), 0),
            **WELLNESS_RULES[key]
        }

    return profile_map


# ─── Inference helpers ────────────────────────────────────────────────────────

def predict_individual(
    attrition_model,
    preprocessor,
    employee_dict: dict,
    numeric_cols: list,
    categorical_cols: list
) -> dict:
    """
    Given a single employee dict, return attrition probability and risk tier.
    """
    row = pd.DataFrame([employee_dict])

    # Ensure all expected columns are present
    for col in numeric_cols:
        if col not in row.columns:
            row[col] = 0
    for col in categorical_cols:
        if col not in row.columns:
            row[col] = "Unknown"

    X = preprocessor.transform(row[numeric_cols + categorical_cols])
    prob = float(attrition_model.predict_proba(X)[0, 1])

    if prob >= 0.7:
        risk_tier = "High Risk"
        tier_color = "#E24B4A"
    elif prob >= 0.4:
        risk_tier = "Medium Risk"
        tier_color = "#EF9F27"
    else:
        risk_tier = "Low Risk"
        tier_color = "#1D9E75"

    return {"probability": round(prob, 4), "risk_tier": risk_tier, "tier_color": tier_color}


def run_whatif(
    attrition_model,
    preprocessor,
    base_employee: dict,
    changed_fields: dict,
    numeric_cols: list,
    categorical_cols: list
) -> dict:
    """
    Return attrition probability for baseline and modified employee.
    """
    baseline = predict_individual(attrition_model, preprocessor, base_employee, numeric_cols, categorical_cols)

    modified = base_employee.copy()
    modified.update(changed_fields)
    modified_result = predict_individual(attrition_model, preprocessor, modified, numeric_cols, categorical_cols)

    delta = modified_result["probability"] - baseline["probability"]
    return {
        "baseline": baseline,
        "modified": modified_result,
        "delta": round(delta, 4),
        "direction": "improved" if delta < 0 else "worsened",
    }


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_models(artifacts: dict, path: str = "models/"):
    """Pickle all model artifacts to disk."""
    os.makedirs(path, exist_ok=True)
    for name, obj in artifacts.items():
        with open(os.path.join(path, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)


def load_models(path: str = "models/") -> dict:
    """Load pickled model artifacts."""
    artifacts = {}
    for fname in os.listdir(path):
        if fname.endswith(".pkl"):
            key = fname.replace(".pkl", "")
            with open(os.path.join(path, fname), "rb") as f:
                artifacts[key] = pickle.load(f)
    return artifacts
