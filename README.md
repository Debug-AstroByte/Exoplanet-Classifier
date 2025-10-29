# 🪐 Kepler Exoplanet Classifier

A deep learning–based web app that classifies **Kepler light curves** into _real exoplanets_ or _false positives_.  
Built with TensorFlow, Streamlit, and NASA’s Kepler dataset.

[![Streamlit App](https://img.shields.io/badge/🚀-Open%20App-brightgreen?style=for-the-badge)](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)
[![Made with Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

---

## 🌌 Overview

The **Kepler Exoplanet Classifier** uses a Convolutional Neural Network (CNN) to identify whether a given Kepler light curve represents a **confirmed planet** or a **false positive**.  
It’s trained using a cleaned subset of NASA’s KOI (Kepler Object of Interest) table.

Key features:
- 🚀 Fast parallel data preprocessing with `ThreadPoolExecutor`
- 📦 Cached local dataset for rapid training
- 🧠 CNN model trained in TensorFlow / Keras
- 🌍 Interactive Streamlit dashboard for exploration & inference
- ⚙️ Runs locally or in the cloud via Streamlit Cloud

---

## 🧩 Repository Structure

```
Exoplanet-Classifier/
├── app.py                     # Streamlit web app
├── process_data.py            # Parallel data processing pipeline
├── Exoplanet.ipynb            # Model training notebook
├── cnn_kepler_200_v2.keras    # Trained CNN model (TensorFlow format)
├── processed_data_output.pkl  # Cached preprocessed dataset
├── kepler_koi_clean.csv       # Cleaned KOI table (optional)
├── requirements.txt           # Dependency list
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
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Generate (or mock) data
```bash
python process_data.py --use-mock
# → creates processed_data_output.pkl in under 10 seconds
```

### 5️⃣ Launch the Streamlit app
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser 🌐

---

## 📊 Model Performance

- **Architecture:** 1D CNN (Convolutional Neural Network)
- **Input:** Binned light curve (400 bins)
- **Output:** Binary classification (Confirmed / False Positive)
- **Training Accuracy:** ~89%
- **Validation AUC:** 0.94

---

## 🛰️ Deployment

This project is deployed live via **Streamlit Cloud**:

👉 [**Launch the App**](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)

All model and dataset files (`.keras`, `.pkl`) are stored within the repo for direct reproducibility.

---

## 🧠 Technologies Used

| Tool | Purpose |
|------|----------|
| **Python 3.11** | Core language |
| **TensorFlow / Keras** | CNN model |
| **Lightkurve** | Kepler data handling |
| **Pandas / NumPy** | Data processing |
| **Streamlit** | Web interface |
| **Joblib** | Serialization of preprocessed data |

---

## 🌠 Acknowledgments

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission Data](https://www.nasa.gov/mission_pages/kepler/)
- [Lightkurve package](https://docs.lightkurve.org/)

---

## 🧑‍🚀 Author

**Debug-AstroByte**  
AI & Astronomy Enthusiast  
📬 [GitHub Profile](https://github.com/Debug-AstroByte)

---

> _“Somewhere, something incredible is waiting to be known.” — Carl Sagan_
