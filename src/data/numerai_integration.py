from numerapi import NumerAPI
import pandas as pd
import logging
import os # For checking file existence
from config.config import INPUT_DIR # To ensure data is downloaded to the correct configured directory

# Configure logger for this module
logger = logging.getLogger(__name__)

# Define the target file path for Numerai training data
DEFAULT_NUMERAI_TRAIN_FILE = "signals/v1.0/train.parquet" # Default as used in example
NUMERAI_TRAIN_FILE_PATH = INPUT_DIR.joinpath(DEFAULT_NUMERAI_TRAIN_FILE)

def download_numerai_data(force_download=False):
    """
    Downloads the Numerai Signals training data if it doesn't already exist or if force_download is True.
    The specific dataset version (e.g., signals/v1.0/train.parquet) should be managed via configuration if needed,
    but here it's using a common default.

    Args:
        force_download (bool): If True, downloads the data even if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing the Numerai training data. Returns None if download or load fails.
    """
    logger.info("Attempting to load or download Numerai Signals training data.")

    # Ensure the directory for the Numerai data exists
    NUMERAI_TRAIN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force_download and os.path.exists(NUMERAI_TRAIN_FILE_PATH):
        logger.info(f"Numerai training data found locally at {NUMERAI_TRAIN_FILE_PATH}. Loading from disk.")
    else:
        logger.info(f"Downloading Numerai Signals training data ({DEFAULT_NUMERAI_TRAIN_FILE}). Force download: {force_download}")
        try:
            napi = NumerAPI() # Default constructor often sufficient for public data
            
            # Optional: List available datasets for user info, but can be verbose
            # try:
            #     available_datasets = napi.list_datasets()
            #     logger.debug(f"Available datasets from NumerAPI: {available_datasets}")
            # except Exception as e:
            #     logger.warning(f"Could not list datasets from NumerAPI: {e}")

            napi.download_dataset(DEFAULT_NUMERAI_TRAIN_FILE, NUMERAI_TRAIN_FILE_PATH)
            logger.info(f"Successfully downloaded Numerai data to {NUMERAI_TRAIN_FILE_PATH}.")
        except Exception as e:
            logger.error(f"Error downloading Numerai data using NumerAPI: {e}")
            return None

    # Load the downloaded data
    try:
        df = pd.read_parquet(NUMERAI_TRAIN_FILE_PATH)
        logger.info(f"Numerai data loaded successfully from {NUMERAI_TRAIN_FILE_PATH}. Shape: {df.shape}")
        if 'friday_date' in df.columns: # Convert friday_date to int if present, as it's often used as int
            df['friday_date'] = df['friday_date'].astype(int)
        return df
    except Exception as e:
        logger.error(f"Error loading Numerai data from parquet file {NUMERAI_TRAIN_FILE_PATH}: {e}")
        return None

def process_numerai_data(df):
    """
    Processes the raw Numerai DataFrame.
    Currently, this function is a placeholder. Implement any necessary cleaning,
    feature engineering specific to Numerai data structure, or type conversions here.

    Args:
        df (pandas.DataFrame): The raw Numerai data DataFrame.

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    if df is None:
        logger.warning("Input DataFrame to process_numerai_data is None. Returning None.")
        return None
        
    logger.info(f"Processing Numerai data. Initial shape: {df.shape}")
    
    # Example processing steps (to be customized):
    # 1. Handle missing values (if any specific strategy is needed for Numerai data)
    #    df.fillna(method='ffill', inplace=True) # Example, not necessarily recommended for all features

    # 2. Feature Engineering (if any specific to Numerai's data structure beyond the main pipeline)
    #    Example: df['custom_numerai_feature'] = df['existing_feature_A'] / df['existing_feature_B']

    # 3. Type conversions (ensure columns are of expected types)
    #    Example: Ensure 'ticker' is string, 'friday_date' is int (already handled in download), target is float.
    #    if 'ticker' in df.columns:
    #        df['ticker'] = df['ticker'].astype(str)
    #    target_col = 'target_20d' # Or get from config
    #    if target_col in df.columns:
    #        df[target_col] = df[target_col].astype(float)
            
    # For this project, primary feature engineering happens in feature_engineering.py using market data.
    # This function is mainly for any pre-processing of Numerai's own data structure before it's merged
    # (e.g., ensuring 'friday_date' and 'ticker' are correctly formatted, selecting specific target columns).
    # Most of this is already handled in main.py's merge logic.
    
    logger.info(f"Numerai data processing complete. Final shape: {df.shape}")
    return df