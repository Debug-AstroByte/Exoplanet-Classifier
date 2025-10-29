# app.py – Streamlit demo
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------
# 1. Load processed data
# -------------------------------------------------
@st.cache_data
def load_data():
    X, y = joblib.load("processed_data_output.pkl")
    return X, y

# -------------------------------------------------
# 2. Load trained model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_kepler_200_v2.keras")

# -------------------------------------------------
# 3. UI
# -------------------------------------------------
st.title("Kepler Exoplanet Classifier")
st.markdown("""
A tiny CNN that decides if a Kepler light-curve is a **real planet** or a **false positive**.
*Data processed with `process_data.py`, model trained in the notebook.*
""")

if st.button("Load data & model"):
    with st.spinner("Loading…"):
        X, y = load_data()
        model = load_model()
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.model = model
    st.success(f"Loaded {X.shape[0]} light-curves!")

# -------------------------------------------------
# 4. Show stats
# -------------------------------------------------
if "X" in st.session_state:
    X, y = st.session_state.X, st.session_state.y
    st.write(f"**Samples:** {X.shape[0]} | **Bins per curve:** {X.shape[1]}")
    st.bar_chart(pd.Series(y).value_counts())

    # -------------------------------------------------
    # 5. Run inference
    # -------------------------------------------------
    if st.button("Run inference on test set"):
        model = st.session_state.model
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        with st.spinner("Predicting…"):
            probs = model.predict(X_test).ravel()
            preds = (probs > 0.5).astype(int)

        # ROC & PR curves
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(rec, prec)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
            ax.plot([0,1],[0,1],"--",color="gray")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
            st.pyplot(fig)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["False", "Confirmed"]).plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

        # Top-6 predictions
        st.subheader("Top-6 confident light curves")
        top_idx = np.argsort(probs)[-6:][::-1]
        for i in top_idx:
            fig, ax = plt.subplots(figsize=(8,2))
            ax.plot(X_test[i].squeeze())
            ax.set_title(f"Pred: {probs[i]:.3f} | True: {y_test[i]}")
            st.pyplot(fig)
