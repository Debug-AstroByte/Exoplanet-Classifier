# app.py – Streamlit demo
import streamlit as st
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_kepler_200_v2.keras")

@st.cache_data
def load_test_split():
    """Load the test split saved by the notebook at training time."""
    try:
        X_test, y_test = joblib.load("test_split.pkl")
        return X_test, y_test
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_full_dataset():
    """Used only for the dataset overview chart."""
    data = np.load("kepler_200_dataset.npz")
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Kepler Exoplanet Classifier")
st.markdown(
    "A 1D CNN that classifies Kepler light-curves as **Confirmed planet** "
    "or **False positive**. Model trained in `Exoplanet.ipynb`."
)

# Dataset overview
with st.expander("Dataset overview", expanded=False):
    X_tr, y_tr, X_te, y_te = load_full_dataset()
    total = len(y_tr) + len(y_te)
    st.write(f"**Total samples:** {total} | **Bins per curve:** {X_tr.shape[1]}")
    import pandas as pd
    counts = pd.Series(
        np.concatenate([y_tr, y_te]), dtype=int
    ).value_counts().rename(index={0: "False Positive", 1: "Confirmed"})
    st.bar_chart(counts)

# Inference
st.subheader("Run inference on held-out test set")

if st.button("Load model & evaluate"):
    model = load_model()
    X_test, y_test = load_test_split()

    if X_test is None:
        st.error(
            "`test_split.pkl` not found. "
            "Re-run the notebook (Cell 2 saves it automatically)."
        )
        st.stop()

    with st.spinner("Predicting…"):
        probs = model.predict(X_test, verbose=0).ravel()
        preds = (probs > 0.5).astype(int)

    # Metrics text
    st.text(classification_report(y_test, preds,
                                   target_names=["False Positive", "Confirmed"]))

    # ROC + PR
    fpr, tpr, _  = roc_curve(y_test, probs)
    prec, rec, _ = precision_recall_curve(y_test, probs)
    roc_auc_val  = auc(fpr, tpr)
    pr_auc_val   = auc(rec, prec)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(rec, prec, label=f"PR AUC = {pr_auc_val:.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds),
        display_labels=["False Positive", "Confirmed"],
    ).plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
    plt.close(fig)

    # Top-6 most confident planet predictions
    st.subheader("Top-6 confident planet predictions")
    for i in np.argsort(probs)[-6:][::-1]:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(X_test[i].squeeze())
        ax.set(
            title=f"Confidence: {probs[i]:.3f} | True: "
                  f"{'Confirmed' if y_test[i] == 1 else 'False Positive'}",
            xlabel="Phase bin", ylabel="Norm. flux",
        )
        st.pyplot(fig)
        plt.close(fig)