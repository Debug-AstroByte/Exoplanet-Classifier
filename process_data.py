import os
import logging
import argparse
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
import joblib
from scipy.signal import convolve
from scipy import interpolate
from astropy.utils.exceptions import AstropyWarning
import lightkurve as lk

# Config
N_SAMPLES_PER_CLASS = 1500
N_BINS = 400
LOG_FILE_PATH = './parallel_processing_script.log'
CACHE_DIR = './kepler_cache'
OUTPUT_NPZ = 'kepler_200_dataset.npz'
OUTPUT_PKL = 'processed_data_output.pkl'
TIMEOUT_SECONDS = 300
MAX_WORKERS = 8

warnings.filterwarnings('ignore', category=AstropyWarning)
os.environ['LIGHTKURVE_CACHE_DIR'] = CACHE_DIR


# Helpers
def _reset_cache(cache_dir):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)


def preprocess_light_curve(row, n_bins, use_mock=False):
    disposition = row['koi_disposition']
    if disposition == 'CANDIDATE':
        return None, None

    label = 1 if disposition == 'CONFIRMED' else 0
    kic_id = row['kepid']
    period = row['koi_period']
    epoch = row['koi_time0bk']

    if use_mock:
        rng = np.random.default_rng(seed=int(kic_id) % 2**31)
        flux_binned = rng.random(n_bins) * 0.1 + 0.95
        return flux_binned.astype('float32'), label

    try:
        lc_collection = lk.search_lightcurve(
            f'KIC {kic_id}', quarter='all', cadence='long'
        ).download_all()

        if not lc_collection:
            return None, None

        lc = lc_collection.stitch().remove_nans()
        lc_clean = lc.remove_outliers(sigma=5).flatten(window_length=75)
        lc_fold = lc_clean.fold(period=period, t0=epoch)

        x = lc_fold.phase.value
        y = lc_fold.flux.value

        if len(x) < 20:
            return None, None

        y = y / np.median(y)
        y = (y - y.mean()) / (y.std() + 1e-8)

        phase_bins = np.linspace(-0.5, 0.5, n_bins)
        interp_func = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        flux_binned = interp_func(phase_bins)

        flux_binned = convolve(flux_binned, np.ones(3) / 3.0, mode='same')

        return flux_binned.astype('float32'), label

    except Exception:
        return None, None


# Pipeline
def run_pipeline(df, n_bins, use_mock):
    df_in = df[['kepid', 'koi_period', 'koi_time0bk', 'koi_disposition']].reset_index(drop=True)
    X_list, y_list = [], []

    _reset_cache(CACHE_DIR)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(preprocess_light_curve, row, n_bins, use_mock): idx
            for idx, row in df_in.iterrows()
        }

        for future in as_completed(futures):
            try:
                flux, label = future.result(timeout=TIMEOUT_SECONDS)
                if flux is not None:
                    X_list.append(flux)
                    y_list.append(label)
            except TimeoutError:
                pass
            except Exception:
                pass

    if not X_list:
        return

    X = np.expand_dims(np.array(X_list, dtype='float32'), -1)
    y = np.array(y_list, dtype='int64')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    np.savez_compressed(OUTPUT_NPZ,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)

    joblib.dump((X, y), OUTPUT_PKL)


# Main
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, mode='w'),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()

    try:
        full_df = pd.read_csv('./kepler_koi_clean.csv')

        confirmed_df = full_df[full_df['koi_disposition'] == 'CONFIRMED']
        fp_df = full_df[full_df['koi_disposition'] == 'FALSE POSITIVE']

        confirmed_sample = confirmed_df.sample(
            n=min(N_SAMPLES_PER_CLASS, len(confirmed_df)),
            random_state=42
        ).dropna(subset=['koi_period', 'koi_time0bk'])

        fp_sample = fp_df.sample(
            n=min(N_SAMPLES_PER_CLASS, len(fp_df)),
            random_state=42
        ).dropna(subset=['koi_period', 'koi_time0bk'])

        df_combined = (
            pd.concat([confirmed_sample, fp_sample])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        run_pipeline(df_combined, N_BINS, args.mock)

    except FileNotFoundError:
        logger.error("kepler_koi_clean.csv not found.")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
