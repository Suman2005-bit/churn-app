import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

@st.cache_resource
def train_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    le = LabelEncoder()
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), encoders

model, feature_cols, encoders = train_model()

tab1, tab2 = st.tabs(["🔍 Single Customer", "📁 Bulk CSV Upload"])

# ─── TAB 1: Single Customer Predictor ───
with tab1:
    st.subheader("Predict churn for one customer")
    st.markdown("Fill in the customer details below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges (₹)", 0, 5000, 999)
        total_charges = monthly_charges * tenure

        st.markdown(f"**Total Charges (auto):** ₹{total_charges}")

    if st.button("🔍 Predict Churn", use_container_width=True):
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        df_input = pd.DataFrame([input_data])

        for col in df_input.select_dtypes(include='object').columns:
            if col in encoders:
                df_input[col] = encoders[col].transform(df_input[col])

        prob = model.predict_proba(df_input)[0][1]
        pred = model.predict(df_input)[0]

        st.markdown("---")

        if prob >= 0.7:
            st.error(f"⚠️ High Churn Risk: {prob*100:.1f}%")
        elif prob >= 0.4:
            st.warning(f"⚠️ Medium Churn Risk: {prob*100:.1f}%")
        else:
            st.success(f"✅ Low Churn Risk: {prob*100:.1f}%")

        # Reasons
        st.subheader("📌 Risk Factors")
        reasons = []
        if contract == "Month-to-month":
            reasons.append("📋 Month-to-month contract — easiest to leave")
        if tenure < 6:
            reasons.append("⏱️ Low tenure — still evaluating alternatives")
        if monthly_charges > 3000:
            reasons.append("💸 High monthly charges — price sensitive")
        if internet_service == "Fiber optic":
            reasons.append("📡 Fiber optic users churn more often")
        if online_security == "No":
            reasons.append("🔓 No online security addon")
        if tech_support == "No":
            reasons.append("🛠️ No tech support addon")

        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("- No major risk factors detected")

        # Smart suggestions
        st.subheader("💡 Recommended Actions")
        if prob >= 0.7:
            st.markdown("""
- 🎁 Offer **20% discount** for next 3 months
- 📅 Suggest upgrade to **annual contract**
- 📞 Schedule a **priority support call**
- 🔒 Offer free **Online Security** addon
            """)
        elif prob >= 0.4:
            st.markdown("""
- 📧 Send a **retention email** with loyalty offer
- 🎯 Suggest a **plan upgrade**
            """)
        else:
            st.markdown("""
- ✅ Customer is happy — no action needed
- 💌 Send a **thank you loyalty reward**
            """)

# ─── TAB 2: Bulk CSV ───
with tab2:
    st.subheader("Upload customer CSV for bulk predictions")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        ids = df['customerID'].copy()
        df.drop('customerID', axis=1, inplace=True)

        if 'Churn' in df.columns:
            df.drop('Churn', axis=1, inplace=True)

        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])

        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)

        df['CustomerID'] = ids.values
        df['Churn Probability'] = (probs * 100).round(1)
        df['Prediction'] = ['🔴 Churns' if p == 1 else '🟢 Stays' for p in preds]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Likely to Churn", int(preds.sum()))
        col3.metric("Churn Rate", f"{preds.mean()*100:.1f}%")

        result = df[['CustomerID', 'Churn Probability', 'Prediction']].sort_values(
            'Churn Probability', ascending=False)
        st.dataframe(result, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(probs * 100, bins=20, color='#e74c3c', edgecolor='white')
        ax.set_xlabel('Churn Probability (%)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Churn Risk Distribution')
        st.pyplot(fig)

        csv = result.to_csv(index=False)
        st.download_button("⬇️ Download Results", csv, "churn_predictions.csv", "text/csv")