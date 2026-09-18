# Dataset

The dataset used in this project is the **IBM Watson Telco Customer Churn** dataset.

It's not included in this repo because of file size and Kaggle's terms of use.

## Download

1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in this folder (`data/`)

The notebook expects the file at this path:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## About the dataset

- **Source:** IBM Sample Datasets (via Kaggle)
- **Rows:** 7,043 customers
- **Columns:** 21 features + 1 target (Churn)
- **Target:** Binary — Yes (churned) / No (did not churn)
- **Class split:** 73.5% No Churn, 26.5% Churn

### Feature categories

| Category | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Target | Churn |
