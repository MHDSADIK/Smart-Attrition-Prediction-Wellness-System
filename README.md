# Employee Attrition & Wellness System

An end-to-end ML dashboard for HR decision-making.

## Project Structure

```
attrition_system/
│
├── data_loader.py       # CSV ingestion, validation, preprocessing, feature engineering
├── model_trainer.py     # Attrition classifier + wellness clustering + inference helpers
├── app.py               # Streamlit dashboard (5 tabs)
├── requirements.txt     # Python dependencies
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

1. Upload your HR CSV in the sidebar
2. Adjust model settings (K-Means clusters, test split %)
3. Click **Train Models**
4. Explore the 5 tabs:

| Tab | What you get |
|---|---|
| 📊 Overview | KPI cards, attrition charts, model performance metrics |
| 🎯 Individual Risk | Per-employee risk score, gauge chart, SHAP explanations |
| 🔄 What-If | Drag sliders to simulate HR interventions |
| 👥 Cohort | Risk breakdown by department, job role, age group |
| 💡 Wellness | K-Means clusters with radar charts + HR recommendations |

## Required CSV Columns

| Column | Type | Values |
|---|---|---|
| Age | numeric | 18–65 |
| Attrition | text/int | Yes/No or 1/0 |
| Department | text | any |
| JobSatisfaction | int | 1–4 |
| MonthlyIncome | numeric | any |
| OverTime | text | Yes/No |
| YearsAtCompany | numeric | any |
| WorkLifeBalance | int | 1–4 |

> The IBM HR Analytics dataset (1470 rows) works perfectly out of the box.
> Download: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Optional Columns (enrich the model)

`BusinessTravel`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`,
`Gender`, `JobInvolvement`, `JobLevel`, `JobRole`, `MaritalStatus`,
`NumCompaniesWorked`, `PerformanceRating`, `RelationshipSatisfaction`,
`StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`,
`YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`

## Models Used

- **Attrition**: XGBoost (with GradientBoosting fallback) + CalibratedClassifierCV
- **Wellness**: KMeans clustering + rule-based profile assignment
- **Explainability**: SHAP TreeExplainer / KernelExplainer

## Feature Engineering (auto-applied)

| Feature | Formula |
|---|---|
| IncomePerYear | MonthlyIncome / Age |
| CompanyLoyaltyRatio | YearsAtCompany / TotalWorkingYears |
| PromotionLag | YearsAtCompany − YearsSinceLastPromotion |
| WellbeingScore | (JobSatisfaction + WorkLifeBalance) / 2 |
