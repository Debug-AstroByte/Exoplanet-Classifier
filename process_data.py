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

#  Config 
N_SAMPLES_PER_CLASS = 1500
N_BINS = 400
LOG_FILE_PATH = './parallel_processing_script.log'
CACHE_DIR = './kepler_cache'
OUTPUT_NPZ = 'kepler_200_dataset.npz'   # primary output (used by notebook)
OUTPUT_PKL = 'processed_data_output.pkl' # secondary output (legacy / app.py fallback)
TIMEOUT_SECONDS = 300
MAX_WORKERS = 8

warnings.filterwarnings('ignore', category=AstropyWarning)
os.environ['LIGHTKURVE_CACHE_DIR'] = CACHE_DIR

# Helpers 

def _reset_cache(cache_dir, logger):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Cache directory ready.")


def preprocess_light_curve(row, n_bins, use_mock=False):
    """
    Fetch, clean, fold, and bin one light curve.
    Returns (flux_array, label) or (None, None) on skip/failure.
    """
    logger = logging.getLogger(__name__)

    disposition = row['koi_disposition']
    if disposition == 'CANDIDATE':
        return None, None
    label  = 1 if disposition == 'CONFIRMED' else 0
    kic_id = row['kepid']
    period = row['koi_period']
    epoch  = row['koi_time0bk']

    logger.debug(f"Processing KIC {kic_id}: period={period:.4f}, epoch={epoch:.4f}")

    if use_mock:
        rng         = np.random.default_rng(seed=int(kic_id) % 2**31)
        flux_binned = rng.random(n_bins) * 0.1 + 0.95
        return flux_binned.astype('float32'), label

    try:
        lc_collection = lk.search_lightcurve(
            f'KIC {kic_id}', quarter='all', cadence='long'
        ).download_all()

        if not lc_collection:
            logger.warning(f"No light curve found for KIC {kic_id}")
            return None, None

        lc       = lc_collection.stitch().remove_nans()
        lc_clean = lc.remove_outliers(sigma=5).flatten(window_length=75)
        lc_fold  = lc_clean.fold(period=period, t0=epoch)

        x = lc_fold.phase.value
        y = lc_fold.flux.value

        if len(x) < 20:
            logger.warning(f"Too few points ({len(x)}) for KIC {kic_id}")
            return None, None

        # Normalising
        y = y / np.median(y)
        y = (y - y.mean()) / (y.std() + 1e-8)

        # fixed-length grid
        phase_bins  = np.linspace(-0.5, 0.5, n_bins)
        interp_func = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        flux_binned = interp_func(phase_bins)

        # Smoothing
        flux_binned = convolve(flux_binned, np.ones(3) / 3.0, mode='same')

        return flux_binned.astype('float32'), label

    except Exception as e:
        logger.error(f"Failed KIC {kic_id}: {type(e).__name__} – {e}")
        return None, None


# Pipeline 

def run_pipeline(df, n_bins, use_mock, logger):
    df_in  = df[['kepid', 'koi_period', 'koi_time0bk', 'koi_disposition']].reset_index(drop=True)
    X_list, y_list = [], []

    _reset_cache(CACHE_DIR, logger)
    logger.info("Starting processing")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(preprocess_light_curve, row, n_bins, use_mock): idx
            for idx, row in df_in.iterrows()
        }
        for done, future in enumerate(as_completed(futures), 1):
            try:
                flux, label = future.result(timeout=TIMEOUT_SECONDS)
                if flux is not None:
                    X_list.append(flux)
                    y_list.append(label)
            except TimeoutError:
                logger.warning(f"Task {futures[future]} timed out.")
            except Exception as e:
                logger.error(f"Result error: {type(e).__name__} – {e}")

            if done % 100 == 0 or done == len(futures):
                logger.info(f"Progress: {done}/{len(futures)} | Successes: {len(X_list)}")

    _reset_cache(CACHE_DIR, logger)

    if not X_list:
        logger.warning("No samples processed. Outputs not saved.")
        return

    X = np.expand_dims(np.array(X_list, dtype='float32'), -1)  # (N, 400, 1)
    y = np.array(y_list, dtype='int64')

    # Train / test split (consistent with notebook)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Saving .npz (used by notebook) and .pkl (legacy)
    np.savez_compressed(OUTPUT_NPZ,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)
    joblib.dump((X, y), OUTPUT_PKL)

    logger.info(f"Saved {OUTPUT_NPZ}  — train {X_train.shape}, test {X_test.shape}")
    logger.info(f"Saved {OUTPUT_PKL}  — full array {X.shape}")




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
    parser.add_argument('--mock', action='store_true',
                        help='Use synthetic data (no network needed)')
    args = parser.parse_args()

    logger.info(f"use_mock={args.mock}")

    try:
        full_df = pd.read_csv('./kepler_koi_clean.csv')
        logger.info(f"CSV loaded: {full_df.shape}")

        confirmed_df = full_df[full_df['koi_disposition'] == 'CONFIRMED']
        fp_df        = full_df[full_df['koi_disposition'] == 'FALSE POSITIVE']

        
        confirmed_sample = (
            confirmed_df
            .sample(n=min(N_SAMPLES_PER_CLASS, len(confirmed_df)), random_state=42)
            .dropna(subset=['koi_period', 'koi_time0bk'])
        )
        fp_sample = (
            fp_df
            .sample(n=min(N_SAMPLES_PER_CLASS, len(fp_df)), random_state=42)
            .dropna(subset=['koi_period', 'koi_time0bk'])
        )

        df_combined = (
            pd.concat([confirmed_sample, fp_sample])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        logger.info(f"Balanced sample: {len(confirmed_sample)} confirmed + {len(fp_sample)} FP")

        run_pipeline(df_combined, N_BINS, args.mock, logger)

    except FileNotFoundError:
        logger.error("kepler_koi_clean.csv not found.")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)