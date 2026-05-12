# Kepler Exoplanet Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Debug-AstroByte/Exoplanet-Classifier/blob/main/Exoplanet.ipynb)

I wanted to know if a neural network could do what astronomers spend hours doing — look at a star's brightness over time and figure out whether something is actually orbiting it. This is my attempt at that, built as a high schooler with no formal background in either ML or astrophysics.

![Exoplanet](data/pic.jpg)

It's a 1D convolutional neural network trained on phase-folded Kepler light curves. It takes a star's brightness pattern as input and outputs a binary prediction: confirmed exoplanet or false positive. The hardest cases — and the ones the model still struggles with — are diluted eclipsing binaries, which produce dips in brightness that look almost identical to a real planetary transit. More on that in the limitations section.

![Demo](demo.gif)

[Live app](https://exoplanet-classifier-agdeywxg3ngr22rxabzrqu.streamlit.app/)

---

## How it works

The Kepler Space Telescope stared at ~150,000 stars for four years, recording their brightness every 30 minutes. When a planet crosses in front of its star, it blocks a tiny fraction of the light — sometimes as little as 0.01% for an Earth-sized planet — leaving a small periodic dip in the curve. That dip is the signal. Everything else is noise.

The tricky part is that a planet might transit its star dozens of times over four years, with each individual transit buried in noise. Phase-folding solves this: if you know the orbital period, you can stack all the transits on top of each other and average them into a clean signal. The model takes that averaged, 400-bin representation as input rather than the raw time series.

---

## Data

Two sources are used:

**Kaggle Kepler dataset** exoTrain.csv, exoTest.csv — pre-labelled time series from the Kaggle Exoplanet Hunter challenge. Each row is one star, each column one flux measurement, with 3,197 measurements per star. Labels: `2` = confirmed planet, `1` = false positive.

**NASA KOI table** kepler_koi_clean.csv — the Kepler Object of Interest catalogue from the NASA Exoplanet Archive, containing orbital parameters (period, transit epoch, duration) and dispositions for each candidate. Used by the data pipeline to phase-fold raw light curves.

CANDIDATE-labelled stars are excluded from training. Their disposition is unresolved — using them as positive examples would introduce label noise, and the model would be trying to learn from examples where we don't actually know the answer.

---

## Data pipeline (`process_data.py`)

For experiments using raw NASA photometry rather than the Kaggle pre-processed series, process_data.py fetches and processes light curves directly from the MAST archive via lightkurve. This is the slower, more involved path — fetching ~3,000 stars over the network takes a while — but it gives you control over every preprocessing step.

For each star:

1. All available Kepler quarters are downloaded and stitched into one continuous time series.
2. Sigma-clipping removes outliers beyond 5 standard deviations (cosmic rays, momentum dumps).
3. A 75-cadence median filter flattens long-term stellar variability and instrumental drift, leaving only the short transit-timescale signals we care about.
4. The light curve is phase-folded using the known orbital period and transit epoch from the KOI table, then binned to 400 uniformly spaced phase points from −0.5 to +0.5.
5. Each folded curve is normalised by its median, and z-score standardised so all stars live on the same scale.

Fetching is parallelised across 8 threads since the bottleneck is network latency, not CPU. Processed arrays are cached to processed_data_output.pkl, so you only pay that cost once.

kepler_200_dataset.npz is a leftover from an earlier version of the project when I was only working with 200 stars. It's no longer used.

---

## Model

The classifier is a 1D CNN cnn_kepler_200_v2.keras that takes a folded, binned light curve of shape (400, 1) as input.

A 1D CNN made sense here because a transit is a local shape — a dip spanning a contiguous run of phase bins — not a global or sequential pattern. The convolutional filters learn to detect the ingress slope, flat bottom, and egress of the transit profile regardless of small phase shifts, without being told explicitly what to look for.

One thing that mattered a lot in practice: class weights. Confirmed planets are the minority class in the Kaggle dataset, and without correcting for that, the model just learns to predict "false positive" for everything and gets a deceptively high accuracy. Class weights penalise minority-class mistakes more heavily during training, forcing the model to actually pay attention to the planets.

---

## Evaluation

The test set has 565 false positives and 5 confirmed planets, so the class imbalance is severe and the metrics need to be read carefully.

| Metric | Value |
|---|---|
| ROC-AUC | 0.941 |
| PR-AUC | 0.131 |
| Confirmed planet recall | 0.60 |
| Confirmed planet precision | 0.11 |
| Macro F1 | 0.58 |

ROC-AUC looks strong but is inflated by the large true negative pool — worth treating with scepticism on datasets this imbalanced. PR-AUC is the more meaningful number here: 0.131 against a random baseline of ~0.009.

In practice, the model catches 60% of real planets but flags a lot of false alarms — precision of 0.11 means roughly 1 in 9 flagged candidates is actually a planet. For a first-pass screening tool that hands off to further observation, recall matters more than precision, but both numbers are worth knowing.

---

## Limitations

The model only sees phase-folded photometric flux. It has no access to:

- Radial velocity measurements, which reveal companion mass directly
- Centroid shift analysis, which catches background eclipsing binaries that are spatially offset from the target star
- Odd/even eclipse depth comparison, which flags secondary eclipses characteristic of a stellar companion
- Multi-band photometry, which helps distinguish stellar from planetary radii

This means diluted eclipsing binaries are genuinely hard. A background binary that's much fainter than the target produces a shallow, symmetric, periodic dip — geometrically nearly identical to a hot Jupiter transit at this photometric precision. The model gets fooled, and that's not really a fixable problem without additional data sources.

Professional vetting pipelines like Robovetter and vespa combine several of these diagnostics. Adding centroid or secondary eclipse features would be the most meaningful next step for this project.

---

## Web app (`app.py`)

A Streamlit app that loads the saved model and test data, runs inference, and shows the ROC curve, precision-recall curve, confusion matrix, and individual light curve predictions with predicted probabilities.

```bash
streamlit run app.py
```

---

## Getting started

```bash
git clone https://github.com/Debug-AstroByte/Exoplanet-Classifier.git
cd Exoplanet-Classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `exoTrain.csv` and `exoTest.csv` from the [Kaggle Exoplanet Hunter dataset](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data) and place them in the repository root. Then open `Exoplanet.ipynb` and run all cells to train the model. Once complete:

```bash
streamlit run app.py
```

Or skip local setup entirely with the [Colab notebook](https://colab.research.google.com/github/Debug-AstroByte/Exoplanet-Classifier/blob/main/Exoplanet.ipynb).

---

## Project structure

```
├── Exoplanet.ipynb              # Training notebook
├── app.py                       # Streamlit inference app
├── process_data.py              # Raw light curve pipeline (NASA MAST)
├── cnn_kepler_200_v2.keras      # Trained model
├── kepler_koi_clean.csv         # NASA KOI table (needed for process_data.py)
├── exoTrain.csv                 # Kaggle training set (download separately)
├── exoTest.csv                  # Kaggle test set (download separately)
├── kepler_200_dataset.npz       # Legacy dataset, no longer used
└── requirements.txt
```

---

## Requirements

```
tensorflow
lightkurve
scikit-learn
streamlit
joblib
numpy
pandas
matplotlib
```

---

## Acknowledgements

NASA and the Kepler mission for the data. The Kaggle Exoplanet Hunter challenge for the labelled dataset. Shallue & Vanderburg (2018) showed that this approach was worth trying in the first place.
