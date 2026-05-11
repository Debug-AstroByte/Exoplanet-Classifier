# 🪐 Kepler Exoplanet Classifier

A deep learning–based web app that classifies **Kepler light curves** into _confirmed exoplanets_ or _false positives_.  
Built with TensorFlow, Streamlit, and NASA's Kepler dataset.

[![Streamlit App](https://img.shields.io/badge/🚀-Open%20App-brightgreen?style=for-the-badge)](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)
[![Made with Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

---

## 🌌 Overview

The **Kepler Exoplanet Classifier** uses a 1D Convolutional Neural Network (CNN) to identify whether a given Kepler light curve represents a **confirmed planet** or a **false positive**.  
Trained on a cleaned, balanced subset of NASA's KOI (Kepler Object of Interest) table.

Key features:
- 🧠 1D CNN trained in TensorFlow / Keras on phase-folded light curves
- 📦 Pre-processed dataset included (`kepler_200_dataset.npz`) — no data fetching needed
- 🔒 Clean train/val/test split with no data leakage
- 🌍 Interactive Streamlit dashboard for inference and evaluation

---

## 🧩 Repository Structure

```
Exoplanet-Classifier/
├── app.py                     # Streamlit web app
├── process_data.py            # Data pipeline (requires MAST network access)
├── Exoplanet.ipynb            # Model training notebook
├── kepler_200_dataset.npz     # Pre-processed dataset (train/test arrays)
├── cnn_kepler_200_v2.keras    # Trained CNN model
├── best_cnn_kepler.keras      # Best checkpoint saved during training
├── kepler_koi_clean.csv       # Raw KOI table (needed only for process_data.py)
├── requirements.txt
├── requirements-mac.txt
├── runtime.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Debug-AstroByte/Exoplanet-Classifier.git
cd Exoplanet-Classifier
```

### 2️⃣ Set up your environment
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Train the model
Open and run `Exoplanet.ipynb` top to bottom.  
This loads `kepler_200_dataset.npz`, trains the CNN, and saves:
- `cnn_kepler_200_v2.keras` — final model
- `best_cnn_kepler.keras` — best checkpoint
- `test_split.pkl` — held-out test set for the app

### 4️⃣ Launch the Streamlit app
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser 🌐

---

## 🔄 Parallel Data Pipeline

`process_data.py` uses a `ThreadPoolExecutor` with 8 workers to fetch and process Kepler light curves in parallel from NASA's MAST archive. Each worker independently fetches, cleans, phase-folds, and bins a light curve into a 400-point array.

### Verify parallel processing works (no network needed)
```bash
python process_data.py --mock
```
This runs the full parallel pipeline with synthetic data — all 8 workers fire and progress is logged in real time. Output: `kepler_200_dataset.npz`.

### Real data fetch (requires access to mast.stsci.edu)
```bash
python process_data.py
```
Fetches live Kepler light curves from NASA. Takes several hours for 3000 KOIs. Not required — the pre-built `.npz` in the repo is sufficient for training and inference.

> **Note:** Restricted network environments (e.g. dev containers, firewalled cloud VMs) may block `mast.stsci.edu`. Use `--mock` to test the pipeline in those cases.

---

## 📊 Model Architecture & Performance

| Property | Value |
|----------|-------|
| Input | Phase-folded light curve, 400 bins |
| Architecture | 1D CNN — Conv×4, BN, GAP, Dropout(0.5), Dense(1) |
| Output | Binary (Confirmed / False Positive) |
| Validation AUC | ~0.94 |
| Labels | CONFIRMED = 1, FALSE POSITIVE = 0 (CANDIDATE excluded) |

---

## 🛰️ Deployment

Deployed live via **Streamlit Cloud**:  
👉 [**Launch the App**](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)

---

## 🧠 Technologies

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | CNN model |
| Lightkurve | Kepler light curve fetching |
| NumPy / Pandas | Data processing |
| Streamlit | Web interface |
| scikit-learn | Metrics, splitting, class weights |

---

## 🌠 Acknowledgments

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/)
- [Lightkurve](https://docs.lightkurve.org/)

---

## 🧑‍🚀 Author

**Debug-AstroByte** — AI & Astronomy Enthusiast  
📬 [GitHub Profile](https://github.com/Debug-AstroByte)

---

> _"Somewhere, something incredible is waiting to be known." — Carl Sagan_
