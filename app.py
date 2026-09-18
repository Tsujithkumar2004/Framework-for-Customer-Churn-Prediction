import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load saved artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open("models/customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("models/encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model_data["model"], model_data["feature_names"], encoders, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run the notebook first to generate the pkl files.")
        st.stop()

model, feature_names, encoders, scaler = load_artifacts()

# ── Helper functions ───────────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    df["AverageMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    df["ChargeToTenureRatio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    def count_services(row):
        count = 0
        for col in service_cols:
            val = str(row.get(col, "No")).strip().lower()
            if val not in ["no", "no phone service", "no internet service"]:
                count += 1
        return count

    df["ServiceCount"] = df.apply(count_services, axis=1)

    df["CustomerEngagementScore"] = (
        (df["tenure"] / 72) * 0.4 +
        (df["ServiceCount"] / 9) * 0.4 -
        (df["ChargeToTenureRatio"] / df["ChargeToTenureRatio"].max().clip(lower=1)) * 0.2
    )

    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["HasLongContract"] = (df["Contract"].isin(["One year", "Two year"])).astype(int)

    def tenure_group(t):
        if t <= 6:    return "New"
        elif t <= 12: return "Growing"
        elif t <= 24: return "Stable"
        else:         return "Loyal"

    df["TenureGroup"] = df["tenure"].apply(tenure_group)
    return df


def encode_features(df):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                df[col] = 0
    return df


def get_risk_label(prob):
    pct = prob * 100
    if pct <= 30:
        return "🟢 Low Risk", "green", pct
    elif pct <= 70:
        return "🟡 Medium Risk", "orange", pct
    else:
        return "🔴 High Risk", "red", pct


def get_recommendations(raw_input, risk_label):
    recs = []

    if raw_input.get("MonthlyCharges", 0) > 70:
        recs.append(("R1", "💰 High monthly charges detected",
                      "Offer a 15% monthly discount or suggest a bundled plan upgrade."))

    if raw_input.get("tenure", 99) <= 6:
        recs.append(("R2", "🎁 New customer (short tenure)",
                      "Send a Welcome Retention Bonus — 1 month of free service."))

    if str(raw_input.get("TechSupport", "No")).lower() in ["no", "0"]:
        recs.append(("R3", "🔧 No Tech Support subscription",
                      "Offer 3 months of complimentary Tech Support."))

    if raw_input.get("Contract", "") == "Month-to-month":
        recs.append(("R4", "📋 Month-to-month contract",
                      "Incentivise an annual contract — 2 months free on upgrade."))

    if raw_input.get("ServiceCount", 10) < 3:
        recs.append(("R5", "📦 Low service count",
                      "Upsell a bundled service package with a promotional discount."))

    if raw_input.get("CustomerEngagementScore", 1) < 0.3:
        recs.append(("R6", "📣 Low engagement score",
                      "Launch a personalised re-engagement campaign via email or SMS."))

    if "High" in risk_label:
        recs.append(("R7", "🏆 High Risk customer",
                      "Assign a dedicated Loyalty Manager and offer a VIP loyalty reward."))

    if not recs:
        recs.append(("—", "✅ No immediate action needed",
                      "Customer appears stable. Monitor quarterly."))

    return recs


def predict(raw_input):
    input_df = pd.DataFrame([raw_input])
    input_df = engineer_features(input_df)
    input_df = encode_features(input_df)

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]
    prediction = "Churn" if prob >= 0.5 else "No Churn"
    risk_label, risk_color, pct = get_risk_label(prob)

    # SHAP
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    except Exception:
        try:
            explainer = shap.LinearExplainer(model, input_scaled)
            shap_values = explainer.shap_values(input_scaled)
            sv = shap_values[0]
        except Exception:
            sv = np.zeros(len(feature_names))

    top5 = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
    recs = get_recommendations({**raw_input,
                                 "ServiceCount": int(input_df["ServiceCount"].iloc[0]),
                                 "CustomerEngagementScore": float(input_df["CustomerEngagementScore"].iloc[0])},
                                risk_label)

    return {
        "prediction": prediction,
        "probability": prob,
        "probability_pct": pct,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "top5_shap": top5,
        "shap_values": sv,
        "recommendations": recs,
        "input_scaled": input_scaled,
        "feature_names": feature_names,
    }


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction")
st.caption("Enter customer details in the sidebar and click Predict.")

# Sidebar inputs
st.sidebar.header("Customer Details")

gender          = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior          = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
partner         = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents      = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service   = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines  = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet        = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_sec      = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_bk       = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_prot     = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support    = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv    = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_mv    = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract        = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless       = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment         = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
total_charges   = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0,
                                           round(monthly_charges * tenure, 2), step=1.0)

raw = {
    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
    "MultipleLines": multiple_lines, "InternetService": internet,
    "OnlineSecurity": online_sec, "OnlineBackup": online_bk,
    "DeviceProtection": device_prot, "TechSupport": tech_support,
    "StreamingTV": streaming_tv, "StreamingMovies": streaming_mv,
    "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

if st.sidebar.button("🔍 Predict Churn", use_container_width=True):
    result = predict(raw)

    # Top metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", "⚠️ Churn" if result["prediction"] == "Churn" else "✅ No Churn")
    col2.metric("Churn Probability", f"{result['probability_pct']:.1f}%")
    col3.metric("Risk Level", result["risk_label"])

    st.divider()

    left, right = st.columns(2)

    # Gauge
    with left:
        st.subheader("Probability Gauge")
        fig, ax = plt.subplots(figsize=(6, 1.2))
        pct = result["probability_pct"]
        color = result["risk_color"]
        ax.barh([""], [pct], color=color, height=0.5)
        ax.barh([""], [100 - pct], left=pct, color="#e0e0e0", height=0.5)
        ax.axvline(30, color="orange", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(70, color="red",    linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Churn Probability (%)")
        ax.set_title(f"{pct:.1f}%", fontweight="bold")
        ax.text(15, 0, "Low", ha="center", va="center", fontsize=8, color="green")
        ax.text(50, 0, "Medium", ha="center", va="center", fontsize=8, color="orange")
        ax.text(85, 0, "High", ha="center", va="center", fontsize=8, color="red")
        st.pyplot(fig)
        plt.close()

    # SHAP
    with right:
        st.subheader("Top SHAP Feature Drivers")
        features = [x[0] for x in result["top5_shap"]]
        values   = [x[1] for x in result["top5_shap"]]
        colors   = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        bars = ax2.barh(features[::-1], values[::-1], color=colors[::-1])
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("SHAP Value")
        ax2.set_title("Feature Contributions (Red = ↑ Churn, Green = ↓ Churn)")
        for bar, val in zip(bars, values[::-1]):
            ax2.text(val + 0.005 if val >= 0 else val - 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:+.3f}", va="center",
                     ha="left" if val >= 0 else "right", fontsize=8)
        st.pyplot(fig2)
        plt.close()

    st.divider()

    # Recommendations
    st.subheader("💡 Retention Recommendations")
    for rule_id, reason, action in result["recommendations"]:
        with st.container():
            st.info(f"**{rule_id} — {reason}**\n\n{action}")

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn**.")

    st.subheader("How it works")
    st.markdown("""
    1. Enter customer information in the sidebar
    2. Click **Predict Churn**
    3. The model returns a churn probability and risk tier
    4. SHAP values show which features are driving the prediction
    5. The recommendation engine suggests specific retention actions
    """)
