# Kepler Exoplanet Classifier

A 1D convolutional neural network trained on phase-folded Kepler light curves to classify stars as hosting a confirmed exoplanet or a false positive. The central challenge is that the dominant false positive category — diluted eclipsing binaries — produces photometric signals nearly identical to a planetary transit, making single-passband brightness data alone an inherently limited input.

---

## What this project does

The Kepler Space Telescope recorded the brightness of ~150,000 stars every 30 minutes for four years. When a planet crosses in front of its star from Kepler's line of sight, it blocks a small fraction of the starlight — as little as 0.01% for an Earth-sized planet — producing a periodic, U-shaped dip in the light curve. This project builds a classifier that takes those brightness recordings as input and predicts whether the dip is caused by a real planet or a false positive.

---

## Data

Two sources are used:

**Kaggle Kepler dataset** (`exoTrain.csv`, `exoTest.csv`) — pre-labelled time series from the Kaggle Exoplanet Hunter challenge. Each row is one star, each column one flux measurement, with 3,197 measurements per star. Labels: `2` = confirmed planet, `1` = false positive.

**NASA KOI table** (`kepler_koi_clean.csv`) — the full Kepler Object of Interest catalogue from the NASA Exoplanet Archive, containing orbital parameters (period, transit epoch, duration) and dispositions for each candidate system. Used by the data pipeline to phase-fold raw light curves.

CANDIDATE-labelled stars are excluded from training. Their disposition is unresolved — using them as positive examples would introduce label noise that degrades classifier performance.

---

## Data pipeline (`process_data.py`)

For experiments using raw NASA photometry rather than the Kaggle pre-processed series, `process_data.py` fetches and processes light curves directly from the MAST archive via `lightkurve`.

For each star:

1. All available Kepler quarters are downloaded and stitched into a single continuous time series.
2. Sigma-clipping removes outliers beyond 5 standard deviations (cosmic rays, momentum dumps).
3. A 75-cadence median filter flattens long-term stellar variability and instrumental systematics, leaving only transit-timescale signals.
4. The light curve is phase-folded using the known orbital period and transit epoch from the KOI table, then binned to 400 uniformly spaced phase points from −0.5 to +0.5. Binning reduces noise by averaging many observations per phase slot.
5. Each folded curve is normalised by its median and then z-score standardised.

Fetching is parallelised across 8 threads (`ThreadPoolExecutor`) since the bottleneck is network latency, not CPU. Processed arrays are cached to `processed_data_output.pkl` via `joblib`.

`kepler_200_dataset.npz` is a legacy artefact from an earlier, smaller experiment. It is no longer used.

---

## Model

The classifier is a 1D CNN (`cnn_kepler_200_v2.keras`) that takes a phase-folded, binned light curve of shape `(400, 1)` as input.

A 1D CNN is appropriate here because the transit signal is a local shape — a dip spanning a contiguous window of phase bins — rather than a global or sequential pattern. Convolutional filters learn to detect the ingress, flat bottom, and egress of the transit profile regardless of minor phase shifts, without needing to be told what shape to look for.

Class weights are applied during training to counteract the imbalance between confirmed planets (minority) and false positives (majority) in the Kaggle dataset. Without this correction, the model converges toward predicting the majority class and achieves high accuracy while missing nearly all real planets.

---

## Evaluation

The primary metric is **ROC-AUC** rather than accuracy. Because confirmed planets are rare, a classifier that predicts "false positive" for every star achieves high accuracy while being scientifically useless. ROC-AUC measures the probability that the model ranks a randomly chosen confirmed planet above a randomly chosen false positive, independent of the decision threshold. The trained model achieves a validation ROC-AUC of 0.94.

The confusion matrix distinguishes between the two error types: false negatives (real planets classified as false positives) and false positives (non-planets flagged as planet candidates). False negatives are the more costly error — a missed planet candidate is unlikely to receive follow-up observation.

---

## Limitations

The model operates only on phase-folded photometric flux. It has no access to:

- Radial velocity measurements, which directly reveal companion mass
- Centroid shift analysis, which identifies off-target contamination from background eclipsing binaries
- Odd/even eclipse depth comparison, which identifies secondary eclipses characteristic of stellar companions
- Multi-band photometry, which can distinguish stellar from planetary radii

As a result, diluted eclipsing binaries — the dominant false positive category in the KOI table — remain difficult cases. A background binary that is sufficiently faint relative to the target star produces a shallow, symmetric, periodic dip that is geometrically indistinguishable from a hot Jupiter transit at the photometric precision available. This is not a modelling failure; it reflects a fundamental information limit of single-passband transit photometry alone.

Professional vetting pipelines (e.g., Robovetter, vespa) combine several of these additional diagnostics. Extending this classifier with centroid or secondary eclipse features would be a meaningful next step.

---

## Web app (`app.py`)

A Streamlit app that loads the saved model and test data, runs inference, and displays the ROC curve, precision-recall curve, confusion matrix, and individual light curve predictions with their associated probabilities. Run with:

```
streamlit run app.py
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

Light curve data from the NASA/IPAC Infrared Science Archive (MAST). KOI catalogue from the NASA Exoplanet Archive. Labelled dataset from the Kaggle Exoplanet Hunter challenge (W. Bharat).
