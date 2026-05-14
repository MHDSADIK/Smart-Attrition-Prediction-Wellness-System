"""
app.py
======
Streamlit dashboard for the Employee Attrition & Wellness System.

Tabs:
  1. Overview       — KPI cards, attrition distribution charts
  2. Individual     — Per-employee risk score + SHAP waterfall
  3. What-If        — Slider-based scenario simulator
  4. Cohort         — Department / age-group risk breakdown
  5. Wellness       — K-Means cluster profiles + recommendations

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Project modules ────────────────────────────────────────────────────────────
from data_loader import (
    load_and_validate, preprocess, build_preprocessor,
    get_feature_names, get_summary_stats
)
from model_trainer import (
    train_attrition_model, train_wellness_model,
    compute_shap_values, predict_individual,
    run_whatif, WELLNESS_RULES
)


# ════════════════════════════════════════════════════════════════════════════════
# Page config
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HR Attrition & Wellness",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #7F77DD;
        margin-bottom: 8px;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin: 24px 0 12px;
        color: #1a1a2e;
    }
    .recommendation-card {
        background: #f0faf5;
        border-left: 3px solid #1D9E75;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
    }
    div[data-testid="stTab"] button {
        font-size: 15px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# Session state helpers
# ════════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "y": None,
        "X_transformed": None,
        "preprocessor": None,
        "feature_names": None,
        "numeric_cols": None,
        "categorical_cols": None,
        "attrition_results": None,
        "wellness_results": None,
        "shap_vals": None,
        "shap_sample": None,
        "trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ════════════════════════════════════════════════════════════════════════════════
# Sidebar — upload + train
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/64/000000/conference-call.png", width=48)
    st.title("HR Analytics")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your HR CSV",
        type=["csv"],
        help="Must include: Age, Attrition, Department, JobSatisfaction, MonthlyIncome, OverTime, YearsAtCompany, WorkLifeBalance"
    )

    st.markdown("#### Model settings")
    n_clusters = st.slider("Wellness clusters (K-Means)", 3, 8, 5)
    test_size = st.slider("Test split %", 10, 40, 20) / 100

    train_button = st.button("🚀 Train Models", type="primary", use_container_width=True)

    if st.session_state.trained:
        st.success("✅ Models ready")
        res = st.session_state.attrition_results
        if res:
            st.metric("ROC-AUC", res["roc_auc"])
            st.metric("CV AUC (5-fold)", f"{res['cv_auc_mean']} ± {res['cv_auc_std']}")

    st.markdown("---")
    st.markdown("""
    **Required columns:**
    `Age`, `Attrition`, `Department`, `JobSatisfaction`,
    `MonthlyIncome`, `OverTime`, `YearsAtCompany`, `WorkLifeBalance`

    **Attrition values:** `Yes`/`No` or `1`/`0`
    """)


# ════════════════════════════════════════════════════════════════════════════════
# Data loading + training pipeline
# ════════════════════════════════════════════════════════════════════════════════
if uploaded_file and train_button:
    with st.spinner("Loading and validating data..."):
        try:
            df_raw, warnings_list = load_and_validate(uploaded_file)
            st.session_state.df_raw = df_raw

            for w in warnings_list:
                st.sidebar.warning(w)

        except ValueError as e:
            st.error(f"**Data validation failed:** {e}")
            st.stop()

    with st.spinner("Preprocessing and engineering features..."):
        df_clean, y, numeric_cols, categorical_cols = preprocess(df_raw) # returns df, y, numeric_cols, categorical_cols
        preprocessor = build_preprocessor(numeric_cols, categorical_cols) # returns fitted ColumnTransformer(preprocessor)
        X_transformed = preprocessor.fit_transform(df_clean) # returns transformed features ready for modeling
        feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols) # returns list of feature names after transformation(numeric_cols + cat_feature_names)

        st.session_state.df_clean = df_clean
        st.session_state.y = y
        st.session_state.X_transformed = X_transformed
        st.session_state.preprocessor = preprocessor
        st.session_state.feature_names = feature_names
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols

    with st.spinner("Training attrition model..."):
        attrition_results = train_attrition_model(
            X_transformed, y, feature_names, test_size=test_size
        )                                                                   # splitting,choosing model, calibrating, evaluating, cross-val, feature importance extraction
        st.session_state.attrition_results = attrition_results

    with st.spinner("Computing SHAP values (this may take 30s)..."):
        shap_vals, shap_sample, _ = compute_shap_values(
            attrition_results["model"], X_transformed, feature_names
        )                                                                         # returns shap values array, sample of X used for shap, and expected value
        st.session_state.shap_vals = shap_vals
        st.session_state.shap_sample = shap_sample

    with st.spinner("Training wellness clustering model..."):
        wellness_results = train_wellness_model(df_raw, n_clusters=n_clusters)     # select wellness features ,overtime value encoding,impute(missing values),scaler,clustering,map clusters
        st.session_state.wellness_results = wellness_results

    st.session_state.trained = True
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# Landing / not yet trained
# ════════════════════════════════════════════════════════════════════════════════
if not st.session_state.trained:
    st.title("👥 Employee Attrition & Wellness System")
    st.markdown("""
    ### How to get started
    1. **Upload your HR CSV** in the sidebar
    2. Click **Train Models**
    3. Explore all 5 tabs

    ---
    ### What this system does

    | Feature | Description |
    |---|---|
    | 🎯 Individual Risk Scoring | Attrition probability per employee with confidence tier |
    | 🔄 What-If Simulation | Change salary, satisfaction, overtime — see risk change |
    | 📊 Cohort Analysis | Risk breakdown by department, age group, job role |
    | 💡 Wellness Recommendations | K-Means clusters mapped to actionable HR interventions |
    | 🔍 SHAP Explainability | Feature-level explanation for every prediction |

    ### Minimum required CSV columns
    `Age`, `Attrition` (Yes/No or 1/0), `Department`, `JobSatisfaction` (1-4),
    `MonthlyIncome`, `OverTime` (Yes/No), `YearsAtCompany`, `WorkLifeBalance` (1-4)

    > **Tip:** The IBM HR Analytics dataset works out of the box.
    > Download from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
    """)
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# Load state
# ════════════════════════════════════════════════════════════════════════════════
df_raw = st.session_state.df_raw
df_clean = st.session_state.df_clean
y = st.session_state.y
X_transformed = st.session_state.X_transformed
preprocessor = st.session_state.preprocessor
feature_names = st.session_state.feature_names
numeric_cols = st.session_state.numeric_cols
categorical_cols = st.session_state.categorical_cols
attrition_results = st.session_state.attrition_results
wellness_results = st.session_state.wellness_results
shap_vals = st.session_state.shap_vals
shap_sample = st.session_state.shap_sample

stats = get_summary_stats(df_clean, y)

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎯 Individual Risk",
    "🔄 What-If Simulation",
    "👥 Cohort Analysis",
    "💡 Wellness"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Workforce Overview")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees", stats["total_employees"])
    c2.metric("Attrition Count", stats["attrition_count"])
    c3.metric("Attrition Rate", f"{stats['attrition_rate']}%")
    c4.metric("Avg Age", stats["avg_age"] or "N/A")
    c5.metric("Avg Tenure (yrs)", stats["avg_tenure"] or "N/A")

    c6, c7 = st.columns(2)
    if stats["avg_satisfaction"]:
        c6.metric("Avg Job Satisfaction", f"{stats['avg_satisfaction']} / 4")
    if stats["overtime_pct"] is not None:
        c7.metric("Overtime Rate", f"{stats['overtime_pct']}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Attrition pie
    with col1:
        counts = y.value_counts().reset_index()
        counts.columns = ["Attrition", "Count"]
        counts["Label"] = counts["Attrition"].map({1: "Left", 0: "Stayed"})
        fig_pie = px.pie(
            counts, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Left": "#E24B4A", "Stayed": "#1D9E75"},
            title="Attrition Distribution"
        )
        fig_pie.update_traces(textinfo="percent+label", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Department attrition rate
    with col2:
        if "Department" in df_raw.columns:
            dept_df = df_raw.copy()
            dept_df["Attrition_bin"] = y
            dept_rate = (
                dept_df.groupby("Department")["Attrition_bin"]
                .agg(["sum", "count"])
                .reset_index()
            )
            dept_rate["rate"] = dept_rate["sum"] / dept_rate["count"] * 100
            dept_rate.columns = ["Department", "Left", "Total", "Attrition Rate %"]
            fig_dept = px.bar(
                dept_rate.sort_values("Attrition Rate %", ascending=True),
                x="Attrition Rate %", y="Department", orientation="h",
                color="Attrition Rate %",
                color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
                title="Attrition Rate by Department"
            )
            st.plotly_chart(fig_dept, use_container_width=True)

    col3, col4 = st.columns(2)

    # Age distribution
    with col3:
        if "Age" in df_clean.columns:
            fig_age = px.histogram(
                df_clean.assign(Attrition=y.map({1: "Left", 0: "Stayed"})),
                x="Age", color="Attrition",
                color_discrete_map={"Left": "#E24B4A", "Stayed": "#1D9E75"},
                barmode="overlay", opacity=0.7,
                title="Age Distribution by Attrition"
            )
            st.plotly_chart(fig_age, use_container_width=True)

    # Income distribution
    with col4:
        if "MonthlyIncome" in df_clean.columns:
            fig_inc = px.box(
                df_clean.assign(Attrition=y.map({1: "Left", 0: "Stayed"})),
                x="Attrition", y="MonthlyIncome", color="Attrition",
                color_discrete_map={"Left": "#E24B4A", "Stayed": "#1D9E75"},
                title="Monthly Income by Attrition"
            )
            st.plotly_chart(fig_inc, use_container_width=True)

    # Model performance
    st.markdown("---")
    st.subheader("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", attrition_results["roc_auc"])
    m2.metric("CV AUC (5-fold)", attrition_results["cv_auc_mean"])
    m3.metric("Avg Precision", attrition_results["avg_precision"])
    m4.metric("CV Std", f"± {attrition_results['cv_auc_std']}")

    # Feature importances
    col_fi, col_cm = st.columns(2)
    with col_fi:
        fi_df = attrition_results["feature_importances"].head(15)
        fig_fi = px.bar(
            fi_df[::-1], x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Purples",
            title="Top 15 Feature Importances"
        )
        fig_fi.update_layout(showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrix
    with col_cm:
        cm = attrition_results["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Stayed", "Left"], y=["Stayed", "Left"],
            text_auto=True, color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: INDIVIDUAL RISK
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Individual Employee Risk Scoring")

    # Score ALL employees
    all_probs = attrition_results["model"].predict_proba(X_transformed)[:, 1]
    scored_df = df_raw.copy()
    scored_df["Attrition_Actual"] = y.values
    scored_df["Risk_Score"] = np.round(all_probs * 100, 1)
    scored_df["Risk_Tier"] = pd.cut(
        scored_df["Risk_Score"],
        bins=[0, 40, 70, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    # Search / filter
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search = st.text_input("Search by employee index or Department", placeholder="e.g. Sales")
    with col_s2:
        tier_filter = st.selectbox("Filter by Risk Tier", ["All", "High Risk", "Medium Risk", "Low Risk"])

    display_df = scored_df.copy()
    if search:
        mask = display_df.apply(lambda r: search.lower() in str(r).lower(), axis=1)
        display_df = display_df[mask]
    if tier_filter != "All":
        display_df = display_df[display_df["Risk_Tier"] == tier_filter]

    # Styled table
    show_cols = ["Age", "Department", "JobSatisfaction", "MonthlyIncome",
                 "OverTime", "YearsAtCompany", "Risk_Score", "Risk_Tier"]
    show_cols = [c for c in show_cols if c in display_df.columns]

    st.dataframe(
        display_df[show_cols].sort_values("Risk_Score", ascending=False).head(50),
        use_container_width=True,
        hide_index=False
    )

    st.info(f"Showing {min(50, len(display_df))} of {len(display_df)} employees")

    # Drill-down
    st.markdown("---")
    st.subheader("Deep-dive: single employee")
    emp_idx = st.number_input(
        "Enter employee row index (0-based)",
        min_value=0, max_value=len(df_raw) - 1, value=0
    )

    employee_row = df_clean.iloc[emp_idx].to_dict()
    result = predict_individual(
        attrition_results["model"], preprocessor,
        employee_row, numeric_cols, categorical_cols
    )

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Attrition Probability", f"{result['probability']*100:.1f}%")
    col_r2.metric("Risk Tier", result["risk_tier"])
    col_r3.metric(
        "Actual Outcome",
        "Left ✗" if y.iloc[emp_idx] == 1 else "Stayed ✓"
    )

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["probability"] * 100,
        number={"suffix": "%"},
        title={"text": "Attrition Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": result["tier_color"]},
            "steps": [
                {"range": [0, 40], "color": "#e8faf2"},
                {"range": [40, 70], "color": "#fef3e2"},
                {"range": [70, 100], "color": "#fdeaea"},
            ],
            "threshold": {
                "line": {"color": result["tier_color"], "width": 3},
                "thickness": 0.75,
                "value": result["probability"] * 100
            }
        }
    ))
    fig_gauge.update_layout(height=260)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP waterfall
    if shap_vals is not None and emp_idx < len(shap_vals):
        st.subheader("SHAP Feature Contributions")
        shap_row = shap_vals[emp_idx]
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_row
        }).sort_values("shap_value", key=abs, ascending=False).head(12)

        colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in shap_df["shap_value"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["shap_value"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors,
        ))
        fig_shap.update_layout(
            title="Top SHAP contributions (red = increases risk, green = reduces risk)",
            xaxis_title="SHAP value",
            height=400
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("Install `shap` package to see feature-level explanations: `pip install shap`")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: WHAT-IF SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("What-If Scenario Simulator")
    st.markdown("Adjust HR levers and instantly see how attrition risk changes.")

    # Select employee
    wi_idx = st.number_input(
        "Base employee row index",
        min_value=0, max_value=len(df_raw) - 1, value=0,
        key="whatif_idx"
    )
    base_row = df_clean.iloc[wi_idx].to_dict()
    base_result = predict_individual(
        attrition_results["model"], preprocessor,
        base_row, numeric_cols, categorical_cols
    )                                                      #{"probability": round(prob, 4), "risk_tier": risk_tier, "tier_color": tier_color}

    st.markdown(f"**Baseline risk:** `{base_result['probability']*100:.1f}%` — {base_result['risk_tier']}")

    st.markdown("---")
    st.subheader("Adjust HR levers")

    changes = {}
    sim_cols = st.columns(2)

    # Salary slider
    with sim_cols[0]:
        if "MonthlyIncome" in base_row:
            cur_inc = int(base_row.get("MonthlyIncome", 5000))
            new_inc = st.slider(
                "Monthly Income", 1000, 20000, cur_inc, step=500,
                help="Current value highlighted"
            )
            if new_inc != cur_inc:
                changes["MonthlyIncome"] = new_inc

        if "JobSatisfaction" in base_row:
            cur_sat = int(base_row.get("JobSatisfaction", 2))
            new_sat = st.slider("Job Satisfaction (1-4)", 1, 4, cur_sat)
            if new_sat != cur_sat:
                changes["JobSatisfaction"] = new_sat

        if "WorkLifeBalance" in base_row:
            cur_wlb = int(base_row.get("WorkLifeBalance", 2))
            new_wlb = st.slider("Work-Life Balance (1-4)", 1, 4, cur_wlb)
            if new_wlb != cur_wlb:
                changes["WorkLifeBalance"] = new_wlb

    with sim_cols[1]:
        if "OverTime" in base_row:
            cur_ot = int(base_row.get("OverTime", 0))
            new_ot = st.selectbox("Overtime", [0, 1], index=cur_ot,
                                   format_func=lambda x: "Yes" if x == 1 else "No")
            if new_ot != cur_ot:
                changes["OverTime"] = new_ot

        if "YearsAtCompany" in base_row:
            cur_yrs = int(base_row.get("YearsAtCompany", 1))
            new_yrs = st.slider("Years at Company", 0, 40, cur_yrs)
            if new_yrs != cur_yrs:
                changes["YearsAtCompany"] = new_yrs

        if "EnvironmentSatisfaction" in base_row:
            cur_env = int(base_row.get("EnvironmentSatisfaction", 2))
            new_env = st.slider("Environment Satisfaction (1-4)", 1, 4, cur_env)
            if new_env != cur_env:
                changes["EnvironmentSatisfaction"] = new_env

    # Run simulation
    if changes:
        sim_result = run_whatif(
            attrition_results["model"], preprocessor,
            base_row, changes, numeric_cols, categorical_cols
        )

        st.markdown("---")
        col_b, col_m, col_d = st.columns(3)
        col_b.metric(
            "Baseline Risk",
            f"{sim_result['baseline']['probability']*100:.1f}%",
            delta=None
        )
        col_m.metric(
            "Modified Risk",
            f"{sim_result['modified']['probability']*100:.1f}%",
            delta=f"{sim_result['delta']*100:+.1f}%",
            delta_color="inverse"
        )
        direction_emoji = "✅ Risk reduced" if sim_result["direction"] == "improved" else "⚠️ Risk increased"
        col_d.markdown(f"### {direction_emoji}")

        # Before/after comparison bar chart
        comp_df = pd.DataFrame({
            "Scenario": ["Baseline", "Modified"],
            "Risk %": [
                sim_result["baseline"]["probability"] * 100,
                sim_result["modified"]["probability"] * 100
            ]
        })
        fig_comp = px.bar(
            comp_df, x="Scenario", y="Risk %",
            color="Scenario",
            color_discrete_map={
                "Baseline": "#7F77DD",
                "Modified": "#1D9E75" if sim_result["direction"] == "improved" else "#E24B4A"
            },
            title="Risk Comparison: Baseline vs Modified",
            range_y=[0, 100]
        )
        fig_comp.add_hline(y=70, line_dash="dash", line_color="#E24B4A", annotation_text="High Risk threshold")
        fig_comp.add_hline(y=40, line_dash="dash", line_color="#EF9F27", annotation_text="Medium Risk threshold")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Show what changed
        st.markdown("**Changes applied:**")
        for k, v in changes.items():
            orig = base_row.get(k, "N/A")
            st.markdown(f"- `{k}`: {orig} → **{v}**")
    else:
        st.info("Adjust any lever above to simulate the effect on attrition risk.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: COHORT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Cohort Risk Analysis")

    all_probs_4 = attrition_results["model"].predict_proba(X_transformed)[:, 1]
    scored_raw = df_raw.copy()
    scored_raw["Risk_Score"] = all_probs_4
    scored_raw["Attrition_Actual"] = y.values

    group_col = st.selectbox(
        "Group by",
        [c for c in ["Department", "JobRole", "MaritalStatus", "Gender",
                      "BusinessTravel", "Education", "JobLevel"]
         if c in scored_raw.columns],
        index=0
    )

    cohort_df = (
        scored_raw.groupby(group_col)
        .agg(
            Count=("Risk_Score", "count"),
            Mean_Risk=("Risk_Score", "mean"),
            High_Risk_Count=("Risk_Score", lambda x: (x >= 0.7).sum()),
            Actual_Attrition=("Attrition_Actual", "mean")
        )
        .reset_index()
    )
    cohort_df["Mean_Risk_%"] = (cohort_df["Mean_Risk"] * 100).round(1)
    cohort_df["Actual_Attrition_%"] = (cohort_df["Actual_Attrition"] * 100).round(1)
    cohort_df["High_Risk_%"] = (cohort_df["High_Risk_Count"] / cohort_df["Count"] * 100).round(1)

    # Bubble chart: mean risk vs cohort size
    fig_bubble = px.scatter(
        cohort_df,
        x=group_col,
        y="Mean_Risk_%",
        size="Count",
        color="Mean_Risk_%",
        color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
        hover_data=["Count", "High_Risk_Count", "Actual_Attrition_%"],
        title=f"Mean Attrition Risk by {group_col} (bubble size = headcount)",
        range_color=[0, 100]
    )
    fig_bubble.update_xaxes(tickangle=30)
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Heatmap: if two categorical columns available
    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(
            cohort_df.sort_values("Mean_Risk_%", ascending=False),
            x=group_col, y="Mean_Risk_%",
            color="Mean_Risk_%",
            color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
            title=f"Mean Risk Score by {group_col}"
        )
        fig_bar.update_xaxes(tickangle=30)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_bar2 = px.bar(
            cohort_df.sort_values("High_Risk_%", ascending=False),
            x=group_col, y="High_Risk_%",
            color="High_Risk_%",
            color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
            title=f"% High-Risk Employees by {group_col}"
        )
        fig_bar2.update_xaxes(tickangle=30)
        st.plotly_chart(fig_bar2, use_container_width=True)

    # Age bucket analysis
    if "Age" in scored_raw.columns:
        st.markdown("---")
        st.subheader("Age Group Risk Breakdown")
        scored_raw["Age_Group"] = pd.cut(
            scored_raw["Age"],
            bins=[18, 25, 35, 45, 55, 70],
            labels=["18-25", "26-35", "36-45", "46-55", "55+"]
        )
        age_df = (
            scored_raw.groupby("Age_Group", observed=True)
            .agg(Count=("Risk_Score", "count"), Mean_Risk=("Risk_Score", "mean"))
            .reset_index()
        )
        age_df["Mean_Risk_%"] = (age_df["Mean_Risk"] * 100).round(1)
        fig_age = px.bar(
            age_df, x="Age_Group", y="Mean_Risk_%",
            text="Count",
            color="Mean_Risk_%",
            color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
            title="Mean Attrition Risk by Age Group (text = headcount)"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Full cohort table
    st.markdown("---")
    st.subheader("Cohort Summary Table")
    display_cohort = cohort_df[[group_col, "Count", "Mean_Risk_%", "High_Risk_Count", "High_Risk_%", "Actual_Attrition_%"]]
    st.dataframe(
        display_cohort.sort_values("Mean_Risk_%", ascending=False),
        use_container_width=True, hide_index=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: WELLNESS
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Wellness Clusters & Recommendations")

    wres = wellness_results
    profile_map = wres["profile_map"]

    # Summary row
    cluster_sizes = [v["cluster_size"] for v in profile_map.values()]
    total = sum(cluster_sizes)

    st.markdown("### Cluster Overview")
    cols = st.columns(len(profile_map))
    for i, (cluster_id, info) in enumerate(profile_map.items()):
        pct = info["cluster_size"] / total * 100
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center; padding:12px; background:#f8f9fa;
                        border-radius:10px; border-top: 4px solid {info['color']}">
                <div style="font-size:22px; font-weight:700; color:{info['color']}">{pct:.0f}%</div>
                <div style="font-size:13px; font-weight:600">{info['label']}</div>
                <div style="font-size:12px; color:#666">{info['cluster_size']} employees</div>
            </div>
            """, unsafe_allow_html=True)

    # Cluster detail
    st.markdown("---")
    st.subheader("Cluster Details & HR Recommendations")

    for cluster_id, info in profile_map.items():
        with st.expander(
            f"Cluster {cluster_id}: **{info['label']}** — {info['cluster_size']} employees",
            expanded=(info["profile_key"] != "healthy")
        ):
            c1, c2 = st.columns([1, 2])
            with c1:
                fig_radar_data = {
                    "Metric": ["Satisfaction", "Work-Life Balance", "Income (norm)", "Overtime (inv)"],
                    "Value": [
                        info["mean_satisfaction"] / 4,
                        info["mean_wlb"] / 4,
                        min(info["mean_income"] / 15000, 1.0),
                        1 - info["mean_overtime"]
                    ]
                }
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=fig_radar_data["Value"] + [fig_radar_data["Value"][0]],
                    theta=fig_radar_data["Metric"] + [fig_radar_data["Metric"][0]],
                    fill="toself",
                    fillcolor=info["color"],
                    line_color=info["color"],
                    opacity=0.5,
                    name=info["label"]
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    height=220,
                    margin=dict(t=20, b=20, l=30, r=30)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown(f"""
                | Metric | Value |
                |---|---|
                | Satisfaction | {info['mean_satisfaction']} / 4 |
                | Work-Life Balance | {info['mean_wlb']} / 4 |
                | Avg Monthly Income | ${info['mean_income']:,.0f} |
                | Overtime Rate | {info['mean_overtime']*100:.0f}% |
                """)

            with c2:
                st.markdown(f"#### Recommended HR Interventions")
                for rec in info["recommendations"]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        ✦ {rec}
                    </div>
                    """, unsafe_allow_html=True)

    # Cluster distribution chart
    st.markdown("---")
    cluster_plot_df = pd.DataFrame([
        {"Cluster": f"C{cid}: {info['label']}", "Count": info["cluster_size"], "Color": info["color"]}
        for cid, info in profile_map.items()
    ])
    fig_clusters = px.bar(
        cluster_plot_df, x="Cluster", y="Count",
        color="Cluster",
        color_discrete_sequence=cluster_plot_df["Color"].tolist(),
        title="Employee Distribution Across Wellness Clusters"
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

    # Cross-tab: wellness cluster vs attrition
    st.subheader("Cluster × Attrition Cross-tab")
    all_probs_5 = attrition_results["model"].predict_proba(X_transformed)[:, 1]
    wellness_cross = pd.DataFrame({
        "Cluster": wres["cluster_labels"],
        "Attrition": y.values,
        "Risk_Score": all_probs_5
    })
    cross_df = (
        wellness_cross.groupby("Cluster")
        .agg(Total=("Attrition", "count"), Left=("Attrition", "sum"),
             Mean_Risk=("Risk_Score", "mean"))
        .reset_index()
    )
    cross_df["Attrition_Rate_%"] = (cross_df["Left"] / cross_df["Total"] * 100).round(1)
    cross_df["Cluster_Label"] = cross_df["Cluster"].map(
        {cid: info["label"] for cid, info in profile_map.items()}
    )
    fig_cross = px.bar(
        cross_df, x="Cluster_Label", y="Attrition_Rate_%",
        color="Mean_Risk",
        color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
        title="Actual Attrition Rate by Wellness Cluster",
        text="Attrition_Rate_%"
    )
    st.plotly_chart(fig_cross, use_container_width=True)
