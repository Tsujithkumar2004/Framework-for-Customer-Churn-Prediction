# Models

The trained model files are not committed to this repo — they're generated locally when you run the notebook.

## Files that get created here

| File | What it contains |
|---|---|
| `customer_churn_model.pkl` | Trained model (Logistic Regression) + feature names |
| `encoders.pkl` | LabelEncoder objects for each categorical column |
| `scaler.pkl` | Fitted StandardScaler |

## How to generate them

Run all cells in `Customer_Churn_Prediction_Upgraded.ipynb`. The notebook saves these files automatically at the end of the training section.

Once they exist here, you can run `app.py` with:
```bash
streamlit run app.py
```

## Why aren't they committed?

- pkl files can get large depending on model size
- They're generated files — not source code
- Anyone who runs the notebook will get the same results anyway (random_state=42 everywhere)
