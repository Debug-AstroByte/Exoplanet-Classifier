# Kepler Exoplanet Classifier

[![Streamlit App](https://img.shields.io/badge/🚀-Open%20App-brightgreen?style=for-the-badge)](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

I built this because I wanted to see if a neural network could do what astronomers spend hours doing — looking at a star's light curve and figuring out if something is actually orbiting it or if it's just noise.

The model takes a phase-folded Kepler light curve (400 bins) and outputs a yes/no: confirmed planet or false positive. It's a 1D CNN trained on ~3000 KOIs from NASA's dataset, and it hits around 0.94 AUC on the held-out test set.

There's also a Streamlit app where you can load the model and see how it performs — ROC curve, confusion matrix, and the top predictions visualised.

---

## How to run it

```bash
git clone https://github.com/Debug-AstroByte/Exoplanet-Classifier.git
cd Exoplanet-Classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `Exoplanet.ipynb` and run it top to bottom. This trains the model and saves everything the app needs. Then:

```bash
streamlit run app.py
```

---

## The data pipeline

`process_data.py` is what actually pulls light curves from NASA's MAST archive and processes them. It runs 8 parallel workers using `ThreadPoolExecutor` — each one fetches, cleans, folds, and bins a single light curve independently. For 3000 KOIs that's a lot of network calls, so parallelising it made a real difference.

If you want to test the pipeline without fetching real data (MAST can be slow or blocked in some environments):

```bash
python process_data.py --mock
```

This runs the exact same parallel pipeline but with synthetic curves — you can see all 8 workers running in the logs. The pre-built dataset (`kepler_200_dataset.npz`) is already in the repo so you don't need to run this just to train the model.

---

## Files

```
├── Exoplanet.ipynb           training notebook
├── app.py                    Streamlit app
├── process_data.py           data pipeline
├── kepler_200_dataset.npz    preprocessed dataset
├── cnn_kepler_200_v2.keras   trained model
├── best_cnn_kepler.keras     best checkpoint from training
├── kepler_koi_clean.csv      raw KOI table (only needed to rerun pipeline)
└── requirements.txt
```

---

## Dataset

This project uses the Kepler labelled time-series dataset available on Kaggle:

[Kepler Labelled Time Series Data (Kaggle)](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data?utm_source=chatgpt.com)

The `exoTrain.csv` and `exoTest.csv` files used in this project were obtained from the dataset above.

The files are not uploaded to this repository since they exceed GitHub's recommended file size limits. To run the project locally, download the dataset and place both CSV files inside the `data/` folder.

---

## A note on labels

I only used CONFIRMED and FALSE POSITIVE labels. CANDIDATE KOIs are excluded — they haven't been verified either way, so including them as positives would just add noise to the training data.

---

## Live app

👉 [exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)

---

Built by [Debug-AstroByte](https://github.com/Debug-AstroByte)

> "Somewhere, something incredible is waiting to be known." — Carl Sagan
