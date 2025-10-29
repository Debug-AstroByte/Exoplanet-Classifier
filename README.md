# 🌌 Exoplanet Classifier – Kepler Mission Data
**A CNN-based classifier that identifies real exoplanets vs false positives from NASA’s Kepler mission.**
Built with TensorFlow and Streamlit.

 **Live Demo:** [Exoplanet Classifier App](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)
 **GitHub Repo:** [Debug-AstroByte/Exoplanet-Classifier](https://github.com/Debug-AstroByte/Exoplanet-Classifier)

---

##  Features
- Trains a lightweight **Convolutional Neural Network (CNN)** on Kepler light-curve data
- Uses **local preprocessed data** for instant setup (no NASA download delay)
- Visualizes **ROC**, **PR**, and **Confusion Matrix** plots
- Interactive **Streamlit web interface** for inference and visualization

---

##  Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/Debug-AstroByte/Exoplanet-Classifier.git
cd Exoplanet-Classifier

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate or load data (mock data included — no NASA download)
python process_data.py
# → creates processed_data_output.pkl

# 5. Run the web demo
streamlit run app.py
```

---

##  Model Info
- **Architecture:** 1D CNN
- **Framework:** TensorFlow / Keras
- **Training Dataset:** NASA Kepler KOI table
- **Saved Model:** `cnn_kepler_200_v2.h5`

---

##  Example Output
- ROC-AUC and PR-AUC curves
- Confusion matrix for validation
- Top 6 most confident light-curve predictions

---

##  Inspiration
This project explores how deep learning can automate the classification of Kepler exoplanet candidates — inspired by NASA’s search for habitable worlds beyond our solar system.
