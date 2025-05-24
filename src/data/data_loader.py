import yfinance as yf
import pandas as pd
from fredapi import Fred
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

def load_market_data(start_date="2010-01-01", end_date="2023-12-31", as_of_date_str: str = None):
    """
    Loads historical market data for predefined global indices and Tesla stock using Yahoo Finance API.
    Data is loaded up to `as_of_date_str` if provided, otherwise up to `end_date`.

    Args:
        start_date (str): The start date for data retrieval (YYYY-MM-DD).
        end_date (str): The default end date if `as_of_date_str` is not provided (YYYY-MM-DD).
        as_of_date_str (str, optional): The specific end date to load data up to (YYYY-MM-DD).
                                        If None, `end_date` parameter is used. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical market data (OHLCV) for all specified assets,
                          with columns prefixed by asset name (e.g., 'SP500_Close'). Returns an empty
                          DataFrame if all downloads fail.
    """
    end_date_to_use = as_of_date_str if as_of_date_str is not None else end_date
    logger.info(f"Starting market data load from {start_date} to {end_date_to_use}.")
    
    # Define major global stock market indices to fetch
    indices = {
        'DAX': '^GDAXI',        # Germany
        'SP500': '^GSPC',       # USA
        'NASDAQ': '^IXIC',      # USA
        'NIKKEI': '^N225',      # Japan
        'BOVESPA': '^BVSP',     # Brazil
        'SHANGHAI': '000001.SS' # China
    }
    
    all_market_data = pd.DataFrame()

    # Download data for each index
    for name, ticker in indices.items():
        logger.info(f"Downloading data for {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date_to_use, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {name} ({ticker}) up to {end_date_to_use}.")
                continue
            df.columns = [f'{name}_{col}' for col in df.columns] # Prefix columns with asset name
            if all_market_data.empty:
                all_market_data = df
            else:
                all_market_data = all_market_data.join(df, how='outer')
            logger.info(f"Successfully downloaded {name} ({ticker}). Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error downloading data for {name} ({ticker}): {e}")

    # Download data for Tesla (TSLA) as an individual stock example
    logger.info("Downloading data for Tesla (TSLA)...")
    try:
        tesla_df = yf.download('TSLA', start=start_date, end=end_date_to_use, progress=False)
        if not tesla_df.empty:
            tesla_df.columns = [f'Tesla_{col}' for col in tesla_df.columns]
            if all_market_data.empty:
                all_market_data = tesla_df
            else:
                all_market_data = all_market_data.join(tesla_df, how='outer')
            logger.info(f"Successfully downloaded Tesla (TSLA). Shape: {tesla_df.shape}")
        else:
            logger.warning(f"No data returned for Tesla (TSLA) up to {end_date_to_use}.")
    except Exception as e:
        logger.error(f"Error downloading data for Tesla (TSLA): {e}")
    
    if all_market_data.empty:
        logger.warning("Market data loading resulted in an empty DataFrame.")
    else:
        # Sort by date index to ensure chronological order, important for time series
        all_market_data.sort_index(inplace=True)
        logger.info(f"Market data loading complete. Final shape: {all_market_data.shape}")
        
    return all_market_data

def load_economic_indicators(api_key):
    """
    Loads key economic indicators from the FRED API.

    Args:
        api_key (str): Your FRED API key.

    Returns:
        pandas.DataFrame: A DataFrame containing the time series for the specified economic indicators.
                          Returns an empty DataFrame if the API key is invalid or all series fail to load.
    """
    # Determine the end date for FRED API calls
    # If as_of_date_str is None, fred.get_series will fetch up to the latest available data when observation_end is None.
    observation_end_date = as_of_date_str 

    log_msg_date_part = f"up to {observation_end_date}" if observation_end_date else "up to latest available"
    logger.info(f"Starting economic indicators load from FRED {log_msg_date_part}.")

    if not api_key or api_key == 'your_fred_api_key' or api_key == 'dummy_key':
        logger.warning("FRED API key is missing or is a placeholder. Skipping economic indicators load.")
        return pd.DataFrame()
        
    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize FRED API client. Ensure API key is valid. Error: {e}")
        return pd.DataFrame()

    # Define list of economic indicators (Series ID, desired column name)
    # Descriptions from FRED:
    # ICSA: Initial Claims, seasonally adjusted. Weekly.
    # UNRATE: Civilian Unemployment Rate, seasonally adjusted. Monthly.
    # CSUSHPISA: Case-Shiller U.S. National Home Price Index, seasonally adjusted. Monthly.
    # MSPUS: Median Sales Price of Houses Sold for the United States. Quarterly.
    # NAPM: ISM Manufacturing PMI Composite Index (aka Purchasing Managers' Index). Monthly.
    # UMCSENT: University of Michigan: Consumer Sentiment. Monthly.
    economic_indicators_to_fetch = [
        ('ICSA', 'Initial_Claims'), # Changed name for consistency
        ('UNRATE', 'Unemployment_Rate'),
        ('CSUSHPISA', 'Case_Shiller_HPI'),
        ('MSPUS', 'Median_Sales_Price_Houses'),
        ('NAPM', 'ISM_Manufacturing_PMI'),
        ('UMCSENT', 'Consumer_Sentiment')
    ]
    
    all_econ_data = pd.DataFrame()
    
    for series_id, name in economic_indicators_to_fetch:
        logger.info(f"Fetching economic indicator: {name} ({series_id}) {log_msg_date_part}...")
        try:
            # Pass observation_end_date to get_series. If it's None, FRED API returns all data up to latest.
            series_data = fred.get_series(series_id, observation_end=observation_end_date)
            if series_data.empty:
                logger.warning(f"No data returned for economic indicator {name} ({series_id}) with end date {observation_end_date}.")
                continue
            all_econ_data[name] = series_data
            logger.info(f"Successfully fetched {name} ({series_id}). Length: {len(series_data)}")
        except ValueError as ve: # fredapi often raises ValueError for bad series ID or other API issues
            logger.error(f"ValueError fetching {name} ({series_id}): {ve}. This might indicate an invalid series ID, date format, or API issue.")
        except Exception as e:
            logger.error(f"Generic error fetching {name} ({series_id}): {e}")
            
    if all_econ_data.empty:
        logger.warning(f"Economic indicators loading resulted in an empty DataFrame for period ending {observation_end_date if observation_end_date else 'latest'}.")
    else:
        # Economic data can have different frequencies; joining them results in NaNs.
        # Sorting by date index is good practice.
        all_econ_data.sort_index(inplace=True)
        logger.info(f"Economic indicators loading complete {log_msg_date_part}. Final shape: {all_econ_data.shape}")
        
    return all_econ_data