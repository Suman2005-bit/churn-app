import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Upload customer data to predict who is likely to churn.")

@st.cache_resource
def train_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_cols = train_model()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    ids = df['customerID'].copy()
    df.drop('customerID', axis=1, inplace=True)

    if 'Churn' in df.columns:
        df.drop('Churn', axis=1, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Predict
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    df['CustomerID'] = ids.values
    df['Churn Probability'] = (probs * 100).round(1)
    df['Prediction'] = ['🔴 Churns' if p == 1 else '🟢 Stays' for p in preds]

    # Summary
    st.subheader("📈 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Likely to Churn", int(preds.sum()))
    col3.metric("Churn Rate", f"{preds.mean()*100:.1f}%")

    # Results table
    st.subheader("🔍 Predictions")
    result = df[['CustomerID', 'Churn Probability', 'Prediction']].sort_values(
        'Churn Probability', ascending=False)
    st.dataframe(result, use_container_width=True)

    # Chart
    st.subheader("📊 Churn Risk Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(probs * 100, bins=20, color='#e74c3c', edgecolor='white')
    ax.set_xlabel('Churn Probability (%)')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Distribution of Churn Risk')
    st.pyplot(fig)

    # High risk customers
    st.subheader("⚠️ High Risk Customers (>70%)")
    high_risk = result[result['Churn Probability'] > 70]
    st.dataframe(high_risk, use_container_width=True)

    # Download
    st.subheader("⬇️ Download Results")
    csv = result.to_csv(index=False)
    st.download_button("Download CSV", csv, "churn_predictions.csv", "text/csv")