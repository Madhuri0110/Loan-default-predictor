import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sqlite3
import os
import shap

from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay


# ==============================
# LOAD MODEL SAFELY
# ==============================
if not os.path.exists("model.pkl"):
    st.error("❌ Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load("model.pkl")

# ==============================
# DATABASE
# ==============================
def create_connection():
    return sqlite3.connect("loan.db")

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_amnt REAL,
            term INTEGER,
            int_rate REAL,
            installment REAL,
            annual_inc REAL,
            dti REAL,
            delinq_2yrs INTEGER,
            revol_util REAL,
            total_acc INTEGER,
            prediction INTEGER,
            probability REAL
        )
    """)
    conn.commit()
    conn.close()

create_table()

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Loan Default System", layout="wide")

st.title("💰 Loan Default Prediction System ")

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("Input Loan Details")

loan_amnt = st.sidebar.number_input("Loan Amount", 1000, 1000000, 10000)
term = st.sidebar.selectbox("Term", [36, 60])
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 10.0)
installment = st.sidebar.number_input("Installment", 100, 50000, 300)
annual_inc = st.sidebar.number_input("Annual Income", 10000, 10000000, 50000)
dti = st.sidebar.slider("DTI", 0.0, 50.0, 10.0)
delinq_2yrs = st.sidebar.slider("Delinq (2yrs)", 0, 10, 0)
revol_util = st.sidebar.slider("Revolving Util (%)", 0.0, 100.0, 30.0)
total_acc = st.sidebar.slider("Total Accounts", 1, 100, 10)

input_data = np.array([[loan_amnt, term, int_rate, installment,
                        annual_inc, dti, delinq_2yrs,
                        revol_util, total_acc]])

# ==============================
# TABS
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prediction",
    "📊 Model Insights",
    "📈 Analytics",
    "🗄️ History",
    "🧠 Explainability"
])

# ==============================
# TAB 1: PREDICTION
# ==============================
with tab1:
    st.subheader("Prediction Result")

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "🔴 Default" if prediction == 1 else "🟢 No Default"

        col1, col2 = st.columns(2)
        col1.metric("Prediction", result)
        col2.metric("Default Probability", f"{probability:.2f}")

        # Save to DB
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                loan_amnt, term, int_rate, installment,
                annual_inc, dti, delinq_2yrs,
                revol_util, total_acc, prediction, probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (*input_data[0], int(prediction), float(probability)))
        conn.commit()
        conn.close()

# ==============================
# TAB 2: MODEL INSIGHTS
# ==============================
with tab2:
    st.subheader("Feature Importance")

    feature_names = [
        "loan_amnt", "term", "int_rate", "installment",
        "annual_inc", "dti", "delinq_2yrs",
        "revol_util", "total_acc"
    ]

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    st.pyplot(fig)

# ==============================
# TAB 3: ANALYTICS
# ==============================
with tab3:
    st.subheader("📈 Model Evaluation (Real Data)")

    import joblib

    # Load test data
    X_test, y_test = joblib.load("test_data.pkl")

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

    # ==============================
    # ROC CURVE
    # ==============================
    st.write("### ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")

    st.pyplot(fig1)

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    st.write("### Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax2)

    st.pyplot(fig2)

# ==============================
# TAB 4: HISTORY
# ==============================
with tab4:
    st.subheader("Prediction History")

    conn = create_connection()
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Data", csv, "predictions.csv", "text/csv")

    # Simple analytics
    if not df.empty:
        st.write("Prediction Distribution")
        st.bar_chart(df['prediction'].value_counts())
# ==============================
# TAB 5: SHAP EXPLAINABILITY
# ==============================
with tab5:
    st.subheader("🧠 SHAP Explainability")

    feature_names = [
        "loan_amnt", "term", "int_rate", "installment",
        "annual_inc", "dti", "delinq_2yrs",
        "revol_util", "total_acc"
    ]

    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Create explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    st.write("### Feature Impact on Prediction")

    # ✅ FIX: Select single sample + class
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)

    # ✅ Bar Plot
    st.write("### Feature Importance")
    fig2 = plt.figure()
    shap.plots.bar(shap_values[:, :, 1], show=False)
    st.pyplot(fig2)