# Customer Churn Prediction

A machine learning project to predict customer churn in the telecom sector. Built this as my final year project — goes beyond just predicting churn, it also explains *why* a customer might leave and suggests what to do about it.

The system uses LightGBM / Logistic Regression as the core model, SHAP for explainability, a 3-tier risk scoring system, and a rule-based recommendation engine. Everything is wrapped in a Streamlit web app.

---

## What this does

1. **Predicts** whether a customer will churn (Yes/No + probability score)
2. **Explains** which factors are driving that prediction (SHAP values per customer)
3. **Scores** the risk level — Low, Medium, or High
4. **Recommends** specific retention actions based on the churn reason

---

## Dataset

IBM Watson Telco Customer Churn dataset — 7,043 customers, 21 features covering demographics, services subscribed, and billing info.

Download it from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV file at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` before running the notebook.

---

## Project Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction_Upgraded.ipynb   # main notebook — run this first
├── app.py                                      # streamlit web app
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md           # download instructions for the dataset
│
├── models/
│   └── README.md           # where pkl files go after running the notebook
│
├── outputs/
│   ├── figures/            # plots saved from the notebook
│   └── reports/            # classification reports
│
└── docs/
    └── architecture.png    # system pipeline diagram
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn and download `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Put it inside the `data/` folder.

**5. Run the notebook**

Open `Customer_Churn_Prediction_Upgraded.ipynb` in Jupyter or Google Colab and run all cells. This will train the models and save the pkl files to the `models/` folder.

**6. Run the Streamlit app**
```bash
streamlit run app.py
```

---

## Pipeline Overview

```
Raw CSV
  ↓
Data Cleaning (drop customerID, fix TotalCharges, IQR outlier capping)
  ↓
Feature Engineering (7 new features — AverageMonthlySpend, ServiceCount, etc.)
  ↓
Label Encoding + SMOTE + StandardScaler
  ↓
Model Training (Logistic Regression, Random Forest, XGBoost, LightGBM)
  ↓
Best Model Selected by ROC-AUC (5-fold CV)
  ↓
SHAP Explanations → Risk Score → Retention Recommendations
  ↓
Streamlit Dashboard
```

---

## Results

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression ⭐ | 0.9171 | 0.78 | 0.55 | **0.66** | 0.60 |
| LightGBM | 0.85 | 0.79 | 0.63 | 0.60 | 0.62 |
| Random Forest | 0.84 | 0.78 | 0.61 | 0.59 | 0.60 |
| XGBoost | 0.83 | 0.78 | 0.60 | 0.61 | 0.61 |

Logistic Regression was selected as the final model based on highest CV ROC-AUC and best recall on the churn class (which matters more than precision in this use case — missing a churner is costlier than a false alarm).

---

## Engineered Features

| Feature | How it's calculated |
|---|---|
| AverageMonthlySpend | TotalCharges / tenure |
| ChargeToTenureRatio | MonthlyCharges / (tenure + 1) |
| ServiceCount | Count of active service subscriptions (0–9) |
| CustomerEngagementScore | Weighted combo of tenure, service count, charge ratio |
| IsMonthToMonth | Binary flag for month-to-month contracts |
| HasLongContract | Binary flag for 1-year or 2-year contracts |
| TenureGroup | New / Growing / Stable / Loyal (based on tenure months) |

---

## Recommendation Rules

| Rule | Trigger | Action |
|---|---|---|
| R1 | MonthlyCharges > $70 | Offer 15% discount or plan upgrade |
| R2 | tenure ≤ 6 months | Welcome bonus — 1 month free |
| R3 | No Tech Support | 3 months free tech support |
| R4 | Month-to-month contract | Incentivise annual contract (2 months free) |
| R5 | ServiceCount < 3 | Bundle upsell with promo |
| R6 | EngagementScore < 0.3 | Re-engagement campaign |
| R7 | High Risk (prob > 70%) | Assign loyalty manager + VIP reward |

Multiple rules can fire for one customer at the same time.

---

## Tech Stack

- Python 3.10
- scikit-learn, XGBoost, LightGBM
- imbalanced-learn (SMOTE)
- SHAP
- Streamlit
- Pandas, NumPy, Matplotlib, Seaborn

---

## Notes

- The notebook was developed and tested on Google Colab
- pkl files are not committed to the repo (too large + generated files) — you need to run the notebook to generate them
- The Kaggle dataset also isn't included for the same reason — download link is above

---

## Acknowledgements

Dataset: IBM Watson Telco Customer Churn (via Kaggle)  
Project guide: Dr. Aarthy S.L, VIT Vellore
