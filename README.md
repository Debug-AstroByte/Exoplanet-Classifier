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

## 🔄 Regenerating the Dataset (optional)

`process_data.py` fetches raw light curves from NASA's MAST archive and rebuilds `kepler_200_dataset.npz`.  
This requires an internet connection with access to `mast.stsci.edu`.

```bash
python process_data.py           # real fetch (~hours)
python process_data.py --mock    # synthetic data, instant, no network needed
```

The pre-processed `.npz` in the repo is sufficient for training and inference without running this script.

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
