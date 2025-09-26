# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# ========= Load Model & Preprocessor =========
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("loan_default_model.h5")
    with open("loan_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("📊 Loan Default Prediction App")

# ========= Sidebar =========
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ========= Main Workflow =========
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    model, preprocessor = load_artifacts()

    # Preprocess
    X_processed = preprocessor.transform(df.drop(columns=["Status"], errors="ignore"))

    # Predictions
    probs = model.predict(X_processed).ravel()
    preds = (probs >= 0.5).astype(int)

    # Show Results
    df_results = df.copy()
    df_results["Default_Probability"] = probs
    df_results["Prediction"] = preds

    st.write("### Predictions")
    st.dataframe(df_results.head(20))

    # Download option
    csv_download = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_download,
        file_name="loan_predictions.csv",
        mime="text/csv"
    )

    # If ground truth column exists, evaluate
    if "Status" in df.columns:
        y_true = df["Status"]
        y_prob = probs
        y_pred = preds

        auc = roc_auc_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

        st.write("### Evaluation Metrics")
        st.metric("ROC AUC", f"{auc:.4f}")
        st.write("Confusion Matrix:", cm)
        st.dataframe(pd.DataFrame(report).transpose())

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        ax.plot([0,1],[0,1],'--', color="red")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

else:
    st.info("👈 Please upload a CSV file to start predictions.")
