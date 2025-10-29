# process_data.py
# =========================================================================
#                         DATA PROCESSING SCRIPT
# =========================================================================

import os
import logging
import argparse
# CRITICAL IMPORTS FOR PARALLEL PROCESSING
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import numpy as np
import pandas as pd
import joblib
import lightkurve as lk
from astropy.stats import sigma_clip
from scipy.signal import convolve
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from scipy import interpolate
import warnings
import time
import shutil

# --- Configuration Parameters (Consistent with Notebook) ---
N_SAMPLES_PER_CLASS = 1500
N_BINS = 400
MAX_LEN = 400
LOG_FILE_PATH = './parallel_processing_script.log' # Log file added back
CACHE_DIR = './kepler_cache'
OUTPUT_PATH = 'processed_data_output.pkl'
TIMEOUT_SECONDS = 300 # Wait up to 5 minutes per light curve fetch

# Suppressing Astropy FITS warnings that clutter the output
warnings.filterwarnings('ignore', category=AstropyWarning)

# Setting lightkurve cache path
os.environ['LIGHTKURVE_CACHE_DIR'] = CACHE_DIR

#  Utility Functions and Main Pipeline
# ------------------------------------------------------------------

def clear_cache_directory(cache_dir, logger):
    """Simple function to ensure cache is clean before starting."""
    if os.path.exists(cache_dir):
        logger.info(f"Clearing old cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Cache directory ready.")


def preprocess_light_curve(row, n_bins, max_len, use_mock=False):
    """
    Fetches, cleans, folds, and bins a single light curve.
    
    Returns: Binned flux array and planet label, or None on failure.
    """
    # Using a logger within the function
    logger = logging.getLogger(__name__)
    
    kic_id = row['kepid']
    period = row['koi_period']
    epoch = row['koi_time0bk']
    label = 1 if row['koi_disposition'] in ['CONFIRMED', 'CANDIDATE'] else 0

    logger.info(f"Processing KIC {kic_id}: period={period}, epoch={epoch}")

    if use_mock:
        # Generate mock data for debugging if flag is set
        flux_binned = np.random.rand(n_bins) * 0.1 + 0.95
        return flux_binned, label

    try:
        # 1. Search for light curve data
        lc_collection = lk.search_lightcurve(
            f'KIC {kic_id}', 
            quarter='all', 
            cadence='long'
        ).download_all()

        if not lc_collection:
            logger.warning(f"No light curve found for KIC {kic_id}")
            return None, None

        # 2. Stitch and clean
        lc = lc_collection.stitch().remove_nans()
        lc_cl = lc.remove_outliers(sigma=5).flatten(window_length=75)
        
        # 3. Fold the light curve
        lc_folded = lc_cl.fold(period=period, t0=epoch)
        
        # 4. Bin the folded light curve
        x = lc_folded.phase.value
        y = lc_folded.flux.value
        y = y / np.median(y)
        y = (y - np.mean(y)) / np.std(y)

        
        if len(x) < 20: 
            logger.warning(f"Insufficient data points ({len(x)}) after cleaning for KIC {kic_id}")
            return None, None

        # Interpolate to fixed length
        interp_func = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        phase_bins = np.linspace(-0.5, 0.5, n_bins)
        flux_binned = interp_func(phase_bins)
        
        # Apply a simple smoothing
        window = np.ones(3)/3.0
        flux_binned = convolve(flux_binned, window, mode='same')
        
        return flux_binned, label

    except Exception as e:
        # Logging errors
        logger.error(f"Failed processing KIC {kic_id}: {type(e).__name__} - {e}")
        return None, None


def run_data_pipeline(df, n_bins, max_len, use_mock, logger):
    """
    Runs the light curve processing in parallel using a ThreadPoolExecutor.
    """
    
    df_filtered = df[['kepid', 'koi_period', 'koi_time0bk', 'koi_disposition']].reset_index(drop=True)

    X_list = []
    y_list = []
    
    # --- PARALLEL EXECUTION WITH THREAD POOL ---
    MAX_WORKERS = 8 
    logger.info(f"Starting parallel processing with {MAX_WORKERS} workers...")
    
    clear_cache_directory(CACHE_DIR, logger)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the pool
        future_to_koi = {
            executor.submit(preprocess_light_curve, row, n_bins, max_len, use_mock): row.name
            for _, row in df_filtered.iterrows()
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_koi)):
            try:
                # Retrieve the result, applying a timeout
                result = future.result(timeout=TIMEOUT_SECONDS) 
                
                if result and result[0] is not None:
                    flux, label = result
                    X_list.append(flux)
                    y_list.append(label)
                    
                logger.info(f"Completed {i+1} / {len(future_to_koi)} tasks. Successes: {len(X_list)}")
            
            except TimeoutError:
                logger.warning(f"Task {future_to_koi[future]} timed out after {TIMEOUT_SECONDS}s.")
            except Exception as e:
                logger.error(f"CRITICAL ERROR retrieving result: {type(e).__name__} - {e}")
            
    # Clean up the cache after processing is done
    clear_cache_directory(CACHE_DIR, logger)

    if X_list:
        X_arr = np.expand_dims(np.array(X_list, dtype='float32'), -1)
        y_arr = np.array(y_list, dtype='int64')
        joblib.dump((X_arr, y_arr), OUTPUT_PATH)
        logger.info(f'Pipeline complete. Saved {OUTPUT_PATH}. Final data shape: {X_arr.shape}')
    else:
        logger.warning("No samples processed. Saving empty file.")
        joblib.dump((np.array([]), np.array([])), OUTPUT_PATH)

    return OUTPUT_PATH
# ------------------------------------------------------------------

if __name__ == '__main__':
    
    # --- LOGGING SETUP ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, mode='w'), # Logs to a file
            logging.StreamHandler() # Logs to the console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("--- Starting parallel data processing script ---")
    
    parser = argparse.ArgumentParser(description="Parallel light curve data processing script.")
    parser.add_argument('--use-mock', action='store_true', help="Use mock data instead of real light curve fetching.")
    args = parser.parse_args()
    
    logger.info(f"Script run with use_mock={args.use_mock}")
    
    try:
        # Load the pre-cleaned CSV saved by the notebook
        full_df = pd.read_csv('./kepler_koi_clean.csv')
        logger.info(f"Loaded input CSV with shape {full_df.shape}")
        
        # Filter and sample for balanced classes
        confirmed_df = full_df[full_df['koi_disposition'] == 'CONFIRMED']
        false_positive_df = full_df[full_df['koi_disposition'] == 'FALSE POSITIVE']
        
        confirmed_sample = confirmed_df.head(N_SAMPLES_PER_CLASS).dropna(subset=['koi_period', 'koi_time0bk'])
        false_sample = false_positive_df.sample(n=N_SAMPLES_PER_CLASS, random_state=42).dropna(subset=['koi_period', 'koi_time0bk'])
        
        # Combine and shuffle
        df_combined = pd.concat([confirmed_sample, false_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Sampled {len(confirmed_sample)} confirmed and {len(false_sample)} false positives for processing.")

        run_data_pipeline(df_combined, N_BINS, MAX_LEN, args.use_mock, logger)
        
    except FileNotFoundError:
        logger.error("kepler_koi_clean.csv not found. Please run the notebook's data loading cell first.")
    except Exception as e:
        logger.critical(f"A general error occurred in the main block: {e}", exc_info=True)