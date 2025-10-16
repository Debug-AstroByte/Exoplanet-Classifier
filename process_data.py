# =========================================================================
#                        FINAL ROBUST SCRIPT 
# =========================================================================

# 1. IMPORTS
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib 
import lightkurve as lk
from astropy.stats import sigma_clip
from scipy.signal import convolve
# Add any other necessary imports here...

# 2. CONSTANTS (Must match your notebook's Section 3)
N_BINS = 100
MAX_LEN = 2048
LOG_FILE_PATH = './parallel_processing_script.log'
CACHE_DIR = './kepler_cache' 

# CRITICAL: Forces Lightkurve to use the local cache
os.environ['LIGHTKURVE_CACHE_DIR'] = CACHE_DIR

# 3. HELPER FUNCTION DEFINITIONS (Your original Sections 4-7 go here)

# PLACEHOLDER: Replace this with your actual Section 4 code
def download_lightcurve_cached(kepid):
    """Downloads light curve using lightkurve and local cache."""
    try:
        lcf = lk.search_lightcurvefile(f'KIC {kepid}', mission='Kepler', cadence='long').download()
        if lcf:
            return lcf.PDCSAP_FLUX.remove_outliers(sigma=5)
    except Exception:
        pass
    return None

# PLACEHOLDER: Replace this with your actual Section 5 code
def safe_get_period_and_epoch(row):
    """Safely retrieves period and epoch, handling NaNs."""
    period = row.get('koi_period')
    epoch = row.get('koi_time0bk')
    if pd.isna(period) or pd.isna(epoch):
        return None, None
    return period, epoch

# PLACEHOLDER: Replace this with your actual Section 6 code
def detrend_and_normalize(lc):
    """Detrends, normalizes, and sigma-clips light curve data."""
    if lc is None: return None
    lc_detrend = lc.remove_outliers(sigma=5).normalize()
    lc_detrend = lc_detrend.flatten(window_length=201)
    return lc_detrend.flux.value

# PLACEHOLDER: Replace this with your actual Section 7 code
def fold_and_resample(lc, period, epoch, n_bins):
    """Folds, bins, and normalizes the light curve."""
    try:
        folded_lc = lc.fold(period=period, epoch_time=epoch)
        # Check if the binned result is empty before accessing .flux.value
        binned_lc = folded_lc.bin(time_bin_size=folded_lc.period.value / n_bins)
        if binned_lc is None or binned_lc.flux is None:
             return None
        return binned_lc.flux.value
    except Exception:
        return None


# 4. MAIN EXECUTION FUNCTION
def process_data_script_entrypoint(confirmed_df, false_df):
    
    # --- Logging Setup (CRITICAL FIX: Solves KeyError) ---
    logger = logging.getLogger('ParallelLog')
    logger.setLevel(logging.INFO)

    # 1. Define a filter to safely inject the KEPID placeholder
    class KepidFilter(logging.Filter):
        """Adds a placeholder kepid key if it's missing for non-thread messages."""
        def filter(self, record):
            if not hasattr(record, 'kepid'):
                record.kepid = '---' # Placeholder when kepid is not set
            return True

    if logger.hasHandlers():
        logger.handlers.clear()

    # Use FileHandler to ensure output bypasses the unstable ZMQ kernel I/O
    handler = logging.FileHandler(LOG_FILE_PATH, mode='w')
    # 2. Use the Kepid placeholder in the formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - KIC %(kepid)s: %(message)s')
    handler.setFormatter(formatter)
    
    # Add the filter and handler
    logger.addFilter(KepidFilter())
    logger.addHandler(handler)
    
    # Silence external libraries
    logging.getLogger("lightkurve").setLevel(logging.ERROR) 

    # Initial log message (kepid will be '---')
    logger.info("Starting external parallel processing. Logs directed to parallel_processing_script.log")

    X_list, y_list = [], []
    N_SAMPLES = len(confirmed_df)

    def process_row_safe(row, label):
        """Processes a single row using the thread-safe logger."""
        kepid = int(row['kepid'])
        # The extra dict provides the 'kepid' value used by the formatter
        log_extra = {'kepid': kepid}
        
        lc = download_lightcurve_cached(kepid)
        if lc is None:
            logger.warning("Download failed or returned None.", extra=log_extra)
            return None

        period, epoch = safe_get_period_and_epoch(row)
        if period is None or epoch is None:
            logger.warning("Skipping: Missing essential period/epoch data.", extra=log_extra)
            return None 

        try:
            res = fold_and_resample(lc, period, epoch, n_bins=N_BINS)
            
            if res is not None and len(res) == N_BINS:
                logger.info("Successfully folded and binned.", extra=log_extra)
                return res, label
            else:
                # Fallback to simple detrending if folding fails or gives wrong size
                arr = detrend_and_normalize(lc)
                if arr is None: 
                    logger.error("Detrending fallback failed.", extra=log_extra)
                    return None
                
                # Resample the detrended array to MAX_LEN
                arr_res = np.interp(np.linspace(0,1,MAX_LEN), np.linspace(0,1,len(arr)), arr)
                logger.info("Successfully used detrending fallback.", extra=log_extra)
                return arr_res, label

        except Exception as e:
            logger.error(f"Critical processing error: {type(e).__name__}: {e}", extra=log_extra) 
            return None

    # --- Thread Execution ---
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process_row_safe, row, 1) for _, row in confirmed_df.iterrows()] + \
                  [ex.submit(process_row_safe, row, 0) for _, row in false_df.sample(n=N_SAMPLES, random_state=42).iterrows()]
        
        for i, f in enumerate(as_completed(futures)):
            res = f.result()
            if res: 
                X_list.append(res[0])
                y_list.append(res[1])
            if (i + 1) % 100 == 0:
                 logger.info(f"Progress update: {i + 1} tasks completed. Current successful samples: {len(X_list)}")


    # 5. SAVE THE RESULT
    X_arr = np.expand_dims(np.array(X_list, dtype='float32'), -1)
    y_arr = np.array(y_list, dtype='int64')

    # Save the large processed arrays safely to disk using joblib
    output_path = 'processed_data_output.pkl'
    joblib.dump((X_arr, y_arr), output_path)

    logger.info(f'Processing complete. Saved data shape: {X_arr.shape}')
    
    return output_path


# 6. ENTRY POINT
if __name__ == '__main__':
    # 6a. Load the input data
    try:
        # Assumes kepler_koi_clean.csv is the pre-filtered, clean file
        full_df = pd.read_csv('./kepler_koi_clean.csv')
    except FileNotFoundError:
        print("ERROR: Could not find './kepler_koi_clean.csv'. Ensure the file is in the project root.")
        exit(1)

    # 6b. Apply the same filtering/sampling logic from your notebook's Section 2
    N_SAMPLES_PER_CLASS = 1500 # Use your actual sample size
    
    confirmed_sample = full_df[full_df['koi_disposition'] == 'CONFIRMED']
    false_sample = full_df[full_df['koi_disposition'] == 'FALSE POSITIVE']
    
    # Apply sampling
    confirmed_sample = confirmed_sample.head(N_SAMPLES_PER_CLASS)
    false_sample = false_sample.sample(n=N_SAMPLES_PER_CLASS, random_state=42)
    
    # 6c. Run the main function
    output_path = process_data_script_entrypoint(confirmed_sample, false_sample)
    
    print("----------------------------------------------------------------")
    print(f"| External processing finished. Results saved to: {output_path} |")
    print("----------------------------------------------------------------")