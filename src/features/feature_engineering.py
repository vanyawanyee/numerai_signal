"""
Feature Engineering Module
==========================

This module is responsible for generating a comprehensive set of financial features 
from raw market and economic data. It leverages cuDF and cuPy for GPU acceleration 
to handle large datasets and computationally intensive operations efficiently.

The module defines various types of features, including:
- Technical Indicators (EMA, SMA, RSI, MACD, Bollinger Bands, ROC, ATR, Stochastic Oscillator)
- Time-Based Features (day of week, month, year, cyclical features)
- Interaction Features (product, ratio, difference, sum of feature pairs)
- Polynomial Features
- Rolling Window Statistical Features (mean, std, median, min, max)
- Cross-Asset Features (correlations, spreads, ratios)
- Data Transformations (log, sqrt, square)
- Lagged Features

The main entry point is the `engineer_features` function, which orchestrates the
generation of all features. It takes raw market data and economic indicators as input
and returns a cuDF DataFrame with all engineered features.

GPU acceleration is used extensively via cuDF for DataFrame operations and cuPy for
numerical computations where cuDF might not have a direct equivalent or for performance.
The selection of features for more complex operations (like interactions, polynomials)
is done strategically to manage the combinatorial explosion of features while aiming for
a rich feature set.
"""
import cudf
import numpy as np
import cupy as cp # For GPU accelerated numpy operations
import logging # For logging information and warnings
from itertools import combinations # For interaction and cross-asset features
import pandas as pd # Used for type checking in some places (np.issubdtype is preferred for NumPy/cuPy dtypes)

# Configure logger for this module
logger = logging.getLogger(__name__)

# Import global config for SEED access
from config import config as global_pipeline_config

# --- Global Parameter Lists for Feature Generation ---
# These lists define the various window sizes, periods, and lags used in feature calculations.
# Modifying these lists will directly impact the number and nature of generated features.

PRICE_VOL_WINDOWS = [5, 10, 20, 30, 50, 100, 200] # For EMAs, SMAs, ATR periods, etc.
LAG_PERIODS = list(range(1, 21)) + list(range(30, 101, 10)) # Up to 20, then 30, 40 ... 100. Total 20 + 8 = 28 lags
SHORT_WINDOWS = [5, 10, 20]
LONG_WINDOWS = [50, 100, 200]
RSI_PERIODS = [7, 14, 21, 30]
MACD_FAST_PERIODS = [6, 12, 26]
MACD_SLOW_PERIODS = [13, 26, 52] # Ensure slow > fast
MACD_SIGNAL_PERIODS = [5, 9, 13]
BOLLINGER_WINDOWS = [10, 20, 50]
BOLLINGER_STD_DEV = [1.5, 2, 2.5]
ROC_PERIODS = [5, 10, 20, 30, 60, 90]
ROLLING_STAT_WINDOWS = [5, 10, 20, 30, 50, 100, 200]

# Helper for safe division
def safe_division(numerator, denominator, fill_value=0.0):
    """
    Performs element-wise division and fills NaNs or Infs resulting from division by zero
    or 0/0 with a specified fill_value.
    Uses cuPy for infinity checks if inputs are cuPy arrays, otherwise NumPy.
    """
    result = numerator / denominator
    # Replace infinities (positive or negative) with NaN first
    # Check if result is a cuDF series or cuPy array to use cp.isinf
    if hasattr(result, 'values_host'): # Heuristic for cuDF series/cupy array
        result[cp.isinf(result.values_host)] = cp.nan
    else: # Fallback to numpy for pandas series or numpy arrays
        result[np.isinf(result)] = np.nan
    return result.fillna(fill_value)

def EWMA(series, span):
    """
    Calculates Exponential Moving Average (EMA/EWMA).
    Relies on cuDF's .ewm() method. 
    The `span` parameter is standard for specifying the decay in terms of span.
    `adjust=False` is used for the common EMA calculation method.
    `min_periods=1` allows EMA to be calculated from the start of the series.
    """
    # Note: cuDF's EWM implementation details (e.g., parameter names like `com` vs `span`) 
    # can evolve. This implementation assumes `span` is directly supported.
    # As of cuDF versions around 23.10+, `.ewm(span=...)` is generally available.
    if not isinstance(series, (cudf.Series, pd.Series)): # Add pd.Series for broader compatibility if inputs vary
        logger.warning(f"EWMA input series is not a cuDF or pandas Series (type: {type(series)}). Attempting to proceed.")
    if series.empty:
        logger.warning("EWMA input series is empty. Returning an empty series.")
        return series.copy()
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def calculate_rsi(prices, period=14):
    """
    Calculates Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))
    Where RS = Average Gain / Average Loss over a specified period.
    """
    delta = prices.diff()
    gain = delta.copy()
    gain[delta <= 0] = 0.0
    loss = delta.copy()
    loss[delta > 0] = 0.0
    loss = loss.abs()

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = safe_division(avg_gain, avg_loss, fill_value=cp.inf) # Use cp.inf for inf
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Replace inf RSI with 100 (strong trend), and NaN from 0/0 rs with 50 (neutral)
    rsi[rs == cp.inf] = 100.0
    # If RS was NaN (e.g., from 0/0 if avg_gain and avg_loss are 0, or if fill_value for rs was np.nan), set RSI to a neutral 50.
    rsi = rsi.fillna(50.0) 
    return rsi


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates Moving Average Convergence Divergence (MACD).
    MACD Line: (Fast EMA - Slow EMA)
    Signal Line: EMA of MACD Line
    MACD Histogram: MACD Line - Signal Line
    Returns MACD line, Signal line, and MACD Histogram.
    """
    if prices.empty:
        logger.warning("MACD calculation: Input prices series is empty.")
        # Return empty series of appropriate types or handle as per desired output for empty inputs
        # Assuming cudf.Series as the standard type in this module for consistency
        return cudf.Series([]), cudf.Series([]), cudf.Series([])

    ema_fast = EWMA(prices, span=fast_period)
    ema_slow = EWMA(prices, span=slow_period)
    macd_line = ema_fast - ema_slow 
    signal_line = EWMA(macd_line, span=signal_period) 
    macd_histogram = macd_line - signal_line 
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    """
    Calculates Bollinger Bands.
    Middle Band: N-period simple moving average (SMA)
    Upper Band: Middle Band + (N-period standard deviation of price * num_std_dev)
    Lower Band: Middle Band - (N-period standard deviation of price * num_std_dev)
    Returns Upper Band, Lower Band, and the Basis (Middle Band/SMA).
    """
    if prices.empty:
        logger.warning("Bollinger Bands calculation: Input prices series is empty.")
        return cudf.Series([]), cudf.Series([]), cudf.Series([])
        
    sma = prices.rolling(window=window, min_periods=1).mean() # Middle Band (SMA)
    std_dev = prices.rolling(window=window, min_periods=1).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band, sma

def calculate_roc(prices, period=12):
    """
    Calculates Rate of Change (ROC) as a percentage.
    ROC = [(Current Price - Price N periods ago) / Price N periods ago] * 100
    """
    if prices.empty:
        logger.warning("ROC calculation: Input prices series is empty.")
        return cudf.Series([])
    shifted_prices = prices.shift(period)
    return safe_division(prices - shifted_prices, shifted_prices) * 100 

def calculate_atr(high_prices, low_prices, close_prices, period=14):
    """
    Calculates Average True Range (ATR).
    ATR measures market volatility by decomposing the entire range of an asset price for that period.
    It's typically calculated as an EMA of the True Range.
    True Range = max[(High - Low), abs(High - Previous Close), abs(Low - Previous Close)]
    """
    if high_prices.empty or low_prices.empty or close_prices.empty:
        logger.warning("ATR calculation: Input price series (high, low, or close) is empty.")
        return cudf.Series([])

    tr1 = high_prices - low_prices  # Range of current period: High - Low
    # Ensure previous close is available for tr2 and tr3; .shift(1) will introduce NaNs at the start
    prev_close = close_prices.shift(1)
    tr2 = (high_prices - prev_close).abs() # Absolute difference: current high vs previous close
    tr3 = (low_prices - prev_close).abs()  # Absolute difference: current low vs previous close
    
    # Create a temporary DataFrame to calculate True Range (max of tr1, tr2, tr3)
    # This is a common way to get row-wise max in cuDF/pandas
    tr_df = cudf.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    true_range = tr_df.max(axis=1) 
    
    atr = EWMA(true_range, span=period) 
    return atr

def calculate_stochastic_oscillator(high_prices, low_prices, close_prices, k_period=14, d_period=3):
    """
    Calculates Stochastic Oscillator (%K and %D).
    %K = [(Current Close - Lowest Low over K periods) / (Highest High over K periods - Lowest Low over K periods)] * 100
    %D = D-period Simple Moving Average of %K
    It indicates the position of the current close relative to the high/low range over a period.
    """
    if high_prices.empty or low_prices.empty or close_prices.empty:
        logger.warning("Stochastic Oscillator calculation: Input price series (high, low, or close) is empty.")
        return cudf.Series([]), cudf.Series([])

    lowest_low = low_prices.rolling(window=k_period, min_periods=1).min()   # Lowest low over the k_period
    highest_high = high_prices.rolling(window=k_period, min_periods=1).max() # Highest high over the k_period
    
    # %K calculation
    # fill_value 0.5 (representing 50%) is used if highest_high == lowest_low (denominator is zero),
    # placing %K in the middle of its range as a neutral value.
    percent_k = safe_division(close_prices - lowest_low, highest_high - lowest_low, fill_value=0.5) * 100 
    
    # %D calculation (SMA of %K)
    percent_d = percent_k.rolling(window=d_period, min_periods=1).mean() 
    return percent_k, percent_d

# --- Feature Generation Functions ---

def add_base_ohlcv_features(df, prefix):
    """
    Adds simple OHLCV (Open, High, Low, Close, Volume) transformations for a given asset prefix.
    These include price range, open-close vs high-low ratio, and close price weighted by volume.
    """
    high_col, low_col, open_col, close_col, volume_col = f'{prefix}_High', f'{prefix}_Low', f'{prefix}_Open', f'{prefix}_Close', f'{prefix}_Volume'
    
    if all(c in df.columns for c in [high_col, low_col, close_col]):
        df[f'{prefix}_PriceRange'] = df[high_col] - df[low_col] # Daily price range
        # Open-Close vs High-Low ratio. Can indicate intraday momentum/reversal.
        df[f'{prefix}_OCHLRatio'] = safe_division(df[open_col] - df[close_col], df[high_col] - df[low_col])
    
    if close_col in df.columns and volume_col in df.columns:
         df[f'{prefix}_CloseTimesVolume'] = df[close_col] * df[volume_col] # Close price weighted by volume


def add_expanded_technical_indicators(df, prefix):
    """
    Adds a wide array of technical indicators with varying parameters for a given asset prefix.
    This function iterates through predefined parameter lists (windows, periods, etc.)
    to generate multiple versions of each indicator.
    """
    close_col = f'{prefix}_Close'
    high_col = f'{prefix}_High'
    low_col = f'{prefix}_Low'
    volume_col = f'{prefix}_Volume'

    if close_col not in df.columns:
        logger.warning(f"Column {close_col} not found for prefix {prefix}. Skipping most technical indicators for it.")
        return

    # EMAs and SMAs for Closing Price and Volume
    for window in PRICE_VOL_WINDOWS:
        df[f'{prefix}_EMA_{window}'] = EWMA(df[close_col], span=window)
        df[f'{prefix}_SMA_{window}'] = df[close_col].rolling(window=window, min_periods=1).mean()
        if volume_col in df.columns:
            df[f'{prefix}_Volume_SMA_{window}'] = df[volume_col].rolling(window=window, min_periods=1).mean()
            # logger.debug(f"Generated EMA_{window}, SMA_{window} for {close_col} and Volume_SMA_{window} for {volume_col}") # Too verbose for many windows

    # RSI (Relative Strength Index)
    for period in RSI_PERIODS:
        df[f'{prefix}_RSI_{period}'] = calculate_rsi(df[close_col], period=period)
    # logger.debug(f"Generated RSI indicators for {prefix}")

    # MACD (Moving Average Convergence Divergence)
    for fast_p in MACD_FAST_PERIODS:
        for slow_p in MACD_SLOW_PERIODS:
            if slow_p <= fast_p: continue # Slow EMA must be longer than Fast EMA
            for signal_p in MACD_SIGNAL_PERIODS:
                macd_line, signal_line, macd_hist = calculate_macd(df[close_col], fast_period=fast_p, slow_period=slow_p, signal_period=signal_p)
                df[f'{prefix}_MACD_{fast_p}_{slow_p}_{signal_p}'] = macd_line
                df[f'{prefix}_MACDSignal_{fast_p}_{slow_p}_{signal_p}'] = signal_line
                df[f'{prefix}_MACDHist_{fast_p}_{slow_p}_{signal_p}'] = macd_hist
    # logger.debug(f"Generated MACD indicators for {prefix}")
    
    # Bollinger Bands
    for window in BOLLINGER_WINDOWS:
        for std_dev in BOLLINGER_STD_DEV:
            upper, lower, basis = calculate_bollinger_bands(df[close_col], window=window, num_std_dev=std_dev)
            df[f'{prefix}_BB_Upper_{window}_{std_dev}'] = upper
            df[f'{prefix}_BB_Lower_{window}_{std_dev}'] = lower
            df[f'{prefix}_BB_Basis_{window}_{std_dev}'] = basis # This is the SMA (middle band)
            df[f'{prefix}_BB_Width_{window}_{std_dev}'] = safe_division(upper - lower, basis) # Bandwidth relative to middle band
    # logger.debug(f"Generated Bollinger Bands for {prefix}")

    # Rate of Change (ROC)
    for period in ROC_PERIODS:
        df[f'{prefix}_ROC_{period}'] = calculate_roc(df[close_col], period=period)
    # logger.debug(f"Generated ROC indicators for {prefix}")

    # ATR and Stochastic Oscillator (require High, Low, Close columns)
    if all(c in df.columns for c in [high_col, low_col, close_col]):
        # ATR (Average True Range)
        for period in PRICE_VOL_WINDOWS: 
            df[f'{prefix}_ATR_{period}'] = calculate_atr(df[high_col], df[low_col], df[close_col], period=period)
        # logger.debug(f"Generated ATR indicators for {prefix}")
        
        # Stochastic Oscillator (%K, %D)
        for k_p in RSI_PERIODS: 
             for d_p in [3, 5]: # Common D periods (smoothing for %K)
                k, d = calculate_stochastic_oscillator(df[high_col], df[low_col], df[close_col], k_period=k_p, d_period=d_p)
                df[f'{prefix}_StochK_{k_p}_{d_p}'] = k
                df[f'{prefix}_StochD_{k_p}_{d_p}'] = d
        # logger.debug(f"Generated Stochastic Oscillator indicators for {prefix}")
    else:
        logger.warning(f"High, Low, or Close columns missing for {prefix}. Skipping ATR and Stochastic Oscillator.")
    
    # Volume Change (Percentage) over 1 period
    if volume_col in df.columns:
        df[f'{prefix}_Volume_Change_1'] = safe_division(df[volume_col].diff(1), df[volume_col].shift(1)) * 100


def add_lagged_features(df, columns_to_lag, lag_periods):
    """Creates lagged versions of specified columns in the DataFrame."""
    df_copy = df.copy() # To avoid issues with adding many columns in a loop to the iterated df
    # logger.debug(f"Generating lagged features for {len(columns_to_lag)} columns over {len(lag_periods)} periods.") # Potentially verbose
    for col in columns_to_lag:
        if col in df.columns: # Check if base column exists
            for lag in lag_periods:
                new_col_name = f'{col}_lag_{lag}'
                df_copy[new_col_name] = df[col].shift(lag)
        else:
            logger.warning(f"Column {col} for lagging not found in DataFrame.")
    return df_copy


def add_rolling_statistical_features(df, base_features, windows):
    """
    Calculates rolling window statistics (mean, std, median, min, max) for specified features.
    Skewness and Kurtosis are noted as placeholders as they might require scipy or custom cuDF implementations.
    """
    df_copy = df.copy()
    # logger.debug(f"Generating rolling statistical features for {len(base_features)} base features over {len(windows)} windows.") # Potentially verbose
    for col in base_features:
        if col in df.columns:
            for window in windows:
                # Ensure min_periods is at least 1 and at most window size.
                # This helps in getting a value even if the window is not fully filled at the beginning of the series.
                min_p = max(1, window // 2) if window > 1 else 1 
                rolling_obj = df[col].rolling(window=window, min_periods=min_p)
                
                df_copy[f'{col}_rolling_mean_{window}'] = rolling_obj.mean()
                df_copy[f'{col}_rolling_std_{window}'] = rolling_obj.std()
                df_copy[f'{col}_rolling_median_{window}'] = rolling_obj.median() # Note: median can be slower
                df_copy[f'{col}_rolling_min_{window}'] = rolling_obj.min()
                df_copy[f'{col}_rolling_max_{window}'] = rolling_obj.max()
                
                # Placeholder for Skewness and Kurtosis if needed:
                # These might not be directly available or efficient in cuDF's rolling object.
                # try:
                #     df_copy[f'{col}_rolling_skew_{window}'] = rolling_obj.skew()
                #     df_copy[f'{col}_rolling_kurt_{window}'] = rolling_obj.kurt()
                # except Exception as e: 
                #     logger.debug(f"Could not compute skew/kurt for {col}_rolling_{window}: {e}")
            # logger.debug(f"Finished rolling stats for base feature: {col}") # Potentially verbose
        else:
            logger.warning(f"Column {col} for rolling stats not found.")
    return df_copy

def add_time_based_features(df):
    """
    Extracts time-based features from the DataFrame's DatetimeIndex.
    Features include day of week, month, year, quarter, etc., and cyclical (sin/cos) transformations.
    """
    if not isinstance(df.index, cudf.DatetimeIndex): # Check if index is already DatetimeIndex
        logger.warning("DataFrame index is not DatetimeIndex. Attempting conversion for time-based features.")
        try:
            # Attempt to convert index to datetime if it's not already
            # This assumes the index is convertible (e.g., string dates, integers representing timestamps)
            df.index = cudf.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert DataFrame index to DatetimeIndex: {e}. Skipping time-based features.")
            return df

    df_copy = df.copy()
    # Basic time features
    df_copy['time_day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    df_copy['time_day_of_month'] = df.index.day
    df_copy['time_day_of_year'] = df.index.dayofyear
    # .weekofyear is deprecated in pandas and might be in cuDF. Use .isocalendar().week
    # Ensure the output of isocalendar().week is compatible (e.g. convert to int16 for memory)
    df_copy['time_week_of_year'] = df.index.isocalendar().week.astype(cp.int16) 
    df_copy['time_month_of_year'] = df.index.month
    df_copy['time_quarter_of_year'] = df.index.quarter
    df_copy['time_year'] = df.index.year
    
    # Cyclical features to represent time's circular nature (e.g., December is close to January)
    # These are useful for models that don't inherently understand ordinal relationships in time.
    day_in_year = df.index.dayofyear.astype(cp.float32) # Ensure float for division
    month_in_year = df.index.month.astype(cp.float32)
    day_in_week = df.index.dayofweek.astype(cp.float32)

    df_copy['time_day_of_year_sin'] = cp.sin(2 * cp.pi * day_in_year / 365.25) # Using 365.25 for leap year avg
    df_copy['time_day_of_year_cos'] = cp.cos(2 * cp.pi * day_in_year / 365.25)
    df_copy['time_month_of_year_sin'] = cp.sin(2 * cp.pi * month_in_year / 12.0)
    df_copy['time_month_of_year_cos'] = cp.cos(2 * cp.pi * month_in_year / 12.0)
    df_copy['time_day_of_week_sin'] = cp.sin(2 * cp.pi * day_in_week / 7.0)
    df_copy['time_day_of_week_cos'] = cp.cos(2 * cp.pi * day_in_week / 7.0)
    return df_copy

def add_interaction_features(df, base_cols_for_interaction):
    """
    Generates interaction features (products, ratios, differences, sums) 
    between specified pairs of columns.
    """
    df_copy = df.copy()
    
    # Filter to only include numeric columns that exist in df and are suitable for interaction
    # np.issubdtype helps check if dtype is a subtype of number (int, float)
    numeric_cols = [col for col in base_cols_for_interaction if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    # logger.debug(f"Generating interaction features for {len(numeric_cols)} numeric base columns.") # Potentially verbose

    for col1, col2 in combinations(numeric_cols, 2): # Create pairs of columns
        # Product
        df_copy[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        # Ratio (handle division by zero)
        df_copy[f'{col1}_ratio_{col2}'] = safe_division(df[col1], df[col2])
        # Difference
        df_copy[f'{col1}_diff_{col2}'] = df[col1] - df[col2]
        # Sum
        df_copy[f'{col1}_sum_{col2}'] = df[col1] + df[col2]

    return df_copy

def add_polynomial_features(df, base_cols_for_poly, degrees=[2,3]):
    """Generates polynomial features (squares, cubes, etc.) for specified columns."""
    df_copy = df.copy()
    numeric_cols = [col for col in base_cols_for_poly if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    # logger.debug(f"Generating polynomial features for {len(numeric_cols)} numeric base columns.") # Potentially verbose

    for col in numeric_cols:
        for degree in degrees:
            df_copy[f'{col}_poly_{degree}'] = df[col] ** degree
    return df_copy

def add_cross_asset_features(df, asset_prefixes, windows):
    """
    Calculates cross-asset features like rolling correlations of log returns, 
    and spreads/ratios of closing prices between different assets.
    """
    df_copy = df.copy()
    
    log_return_cols = [f'{p}_LogReturn' for p in asset_prefixes if f'{p}_LogReturn' in df.columns]
    close_price_cols = [f'{p}_Close' for p in asset_prefixes if f'{p}_Close' in df.columns]
    # logger.debug(f"Found {len(log_return_cols)} log return columns and {len(close_price_cols)} close price columns for cross-asset features.") # Potentially verbose

    # Rolling correlations of log returns between different assets
    for col1, col2 in combinations(log_return_cols, 2):
        for window in windows:
            # Ensure min_periods is reasonable
            min_p = max(1, window // 2) if window > 1 else 1
            df_copy[f'corr_{col1}_vs_{col2}_win{window}'] = df[col1].rolling(window=window, min_periods=min_p).corr(df[col2])
    # logger.debug(f"Generated rolling correlation features for {len(log_return_cols)} assets.") # Potentially verbose

    # Spreads and Ratios of closing prices between different assets
    for col1, col2 in combinations(close_price_cols, 2):
        df_copy[f'spread_{col1}_vs_{col2}'] = df[col1] - df[col2]
        df_copy[f'ratio_{col1}_vs_{col2}'] = safe_division(df[col1], df[col2])
    # logger.debug(f"Generated spread and ratio features for {len(close_price_cols)} assets.")
        
    return df_copy

def add_data_transformations(df, cols_to_transform):
    """
    Applies mathematical transformations (log, sqrt, square) to specified columns.
    Handles non-positive values appropriately for log and sqrt transformations.
    """
    df_copy = df.copy()
    numeric_cols = [col for col in cols_to_transform if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    # logger.debug(f"Applying data transformations to {len(numeric_cols)} columns.") # Potentially verbose

    for col in numeric_cols:
        # Log transform: clip values below a small epsilon to prevent log(0) or log(negative).
        df_copy[f'{col}_log'] = cp.log(df[col].clip(lower=1e-9)) 
        # Square root transform: clip negative values to 0 before sqrt.
        df_copy[f'{col}_sqrt'] = cp.sqrt(df[col].clip(lower=0))
        # Square transform
        df_copy[f'{col}_sq'] = df[col] ** 2
    return df_copy


def engineer_features(market_data_pd, economic_indicators_pd):
    """
    Main function to engineer a comprehensive set of features from market and economic data.
    Orchestrates calls to various specialized feature generation functions.

    Args:
        market_data_pd (pd.DataFrame): Pandas DataFrame of raw market data (OHLCV for multiple assets).
        economic_indicators_pd (pd.DataFrame): Pandas DataFrame of raw economic indicators.

    Returns:
        cudf.DataFrame: A cuDF DataFrame containing all engineered features, ready for model training or selection.
    """
    logger.info(f"Starting feature engineering. Initial market data shape: {market_data_pd.shape}, Economic indicators shape: {economic_indicators_pd.shape}")

    # Convert input pandas DataFrames to cuDF DataFrames for GPU acceleration
    logger.debug("Converting pandas DataFrames to cuDF DataFrames...")
    market_data = cudf.from_pandas(market_data_pd)
    if economic_indicators_pd.empty:
        economic_indicators = cudf.DataFrame() 
        logger.debug("Economic indicators DataFrame is empty.")
    else:
        economic_indicators = cudf.from_pandas(economic_indicators_pd)
    logger.debug(f"cuDF market data shape: {market_data.shape}, cuDF economic indicators shape: {economic_indicators.shape}")

    # Ensure DataFrame indices are DatetimeIndex for time-series operations
    logger.debug("Ensuring DatetimeIndex for DataFrames...")
    if not isinstance(market_data.index, cudf.DatetimeIndex):
        market_data.index = cudf.to_datetime(market_data.index)
    if not economic_indicators.empty and not isinstance(economic_indicators.index, cudf.DatetimeIndex):
        economic_indicators.index = cudf.to_datetime(economic_indicators.index)
    
    # Join market data with economic indicators
    logger.debug("Joining market data and economic indicators...")
    if not economic_indicators.empty:
        # Use suffixes to distinguish potentially overlapping column names from different sources
        data = market_data.join(economic_indicators, how='outer', lsuffix='_mkt', rsuffix='_econ')
    else:
        data = market_data
    data = data.sort_index() # Ensure chronological order after join
    logger.debug(f"Joined data shape: {data.shape}")

    # Initial NaN filling: Fill NaNs resulting from joins or irregular series (e.g., economic data)
    # Forward fill first, then backward fill to propagate last known values and then first known values.
    logger.debug("Performing initial ffill/bfill on joined data...")
    data = data.ffill().bfill()

    # --- Core Feature Engineering Steps ---
    # Define asset prefixes for which features will be generated
    asset_prefixes = ['DAX', 'SP500', 'NASDAQ', 'NIKKEI', 'BOVESPA', 'SHANGHAI', 'Tesla']
    all_original_cols = list(data.columns) # Keep track of original columns for later selections

    # 1. Basic OHLCV transformations and Log Returns
    logger.info("Step 1: Generating basic OHLCV features and log returns...")
    for prefix in asset_prefixes:
        close_col = f'{prefix}_Close'
        if close_col in data.columns:
            # Clip close prices to a very small positive number to prevent issues with log(0) or log(negative)
            data[close_col] = data[close_col].clip(lower=1e-9) 
            data[f'{prefix}_LogReturn'] = cp.log(data[close_col] / data[close_col].shift(1)) # Use cuPy log for speed
            add_base_ohlcv_features(data, prefix) # Add features like PriceRange, OCHLRatio, CloseTimesVolume
        else:
            logger.warning(f"Close column {close_col} not found for prefix {prefix} during log return calculation.")
    logger.info("Step 1 finished.")

    # 2. Expanded Technical Indicators for each asset
    logger.info("Step 2: Generating expanded technical indicators...")
    for prefix in asset_prefixes:
        logger.debug(f"Generating technical indicators for asset: {prefix}")
        add_expanded_technical_indicators(data, prefix)
    logger.info("Step 2 finished.")
    
    # Store columns generated after indicators for later transformation selection
    cols_after_indicators = list(data.columns)
    # Identify newly added numerical columns (potential candidates for transformation)
    numerical_cols_for_transformations = [
        c for c in cols_after_indicators 
        if np.issubdtype(data[c].dtype, np.number) and c not in all_original_cols
    ]

    # Add a random indicator for model robustness testing if needed (using cupy)
    # Use the global SEED from the imported config
    if hasattr(global_pipeline_config, 'SEED'):
        cp.random.seed(global_pipeline_config.SEED) 
        logger.debug(f"Set cuPy random seed to global SEED: {global_pipeline_config.SEED}")
    else:
        cp.random.seed(42) # Fallback if global_pipeline_config.SEED is not available
        logger.warning("global_pipeline_config.SEED not found, using fallback seed 42 for Random_Indicator_GPU.")
    data['Random_Indicator_GPU'] = cp.random.randn(len(data))
    logger.debug("Added Random_Indicator_GPU.")

    # 3. Time-Based Features from DatetimeIndex
    logger.info("Step 3: Generating time-based features...")
    data = add_time_based_features(data)
    logger.info("Step 3 finished.")

    # 4. Cross-Asset Features (Correlations, Spreads, Ratios)
    logger.info("Step 4: Generating cross-asset features...")
    data = add_cross_asset_features(data, asset_prefixes, windows=ROLLING_STAT_WINDOWS)
    logger.info("Step 4 finished.")
    
    # Define a manageable subset of base columns for computationally intensive features
    # This selection is CRUCIAL to keep feature count from exploding too uncontrollably.
    # Focus on log returns, key technical indicators, and important original economic values.
    logger.debug("Defining key features for expansion (interactions, polynomials, rolling stats)...")
    key_features_for_expansion = []
    for p in asset_prefixes:
        # Add key features for each asset prefix if they exist
        if f'{p}_LogReturn' in data.columns: key_features_for_expansion.append(f'{p}_LogReturn')
        if f'{p}_RSI_14' in data.columns: key_features_for_expansion.append(f'{p}_RSI_14') # Example specific RSI
        if f'{p}_SMA_20' in data.columns: key_features_for_expansion.append(f'{p}_SMA_20') # Example specific SMA
        # Check if volume column exists before adding volume-based features
        volume_col_name = f'{p}_Volume' 
        if volume_col_name in data.columns and f'{p}_Volume_SMA_20' in data.columns: 
            key_features_for_expansion.append(f'{p}_Volume_SMA_20')
        if all(c in data.columns for c in [f'{p}_High', f'{p}_Low', f'{p}_Close']) and f'{p}_ATR_20' in data.columns: 
            key_features_for_expansion.append(f'{p}_ATR_20')
        
    if not economic_indicators.empty:
        # Add up to the first 5 existing economic indicator columns
        econ_cols_to_add = [col for col in economic_indicators.columns if col in data.columns][:5]
        key_features_for_expansion.extend(econ_cols_to_add)
    
    # Ensure all selected key_features_for_expansion actually exist in the DataFrame
    key_features_for_expansion = [col for col in key_features_for_expansion if col in data.columns]
    logger.debug(f"Selected {len(key_features_for_expansion)} key features for further expansion: {key_features_for_expansion}")


    # 5. Interaction Features (products, ratios, diffs, sums on the selected key features)
    logger.info("Step 5: Generating interaction features...")
    data = add_interaction_features(data, key_features_for_expansion)
    logger.info("Step 5 finished.")

    # 6. Polynomial Features (on the selected key features)
    logger.info("Step 6: Generating polynomial features...")
    data = add_polynomial_features(data, key_features_for_expansion, degrees=[2, 3]) # Quadratic and cubic
    logger.info("Step 6 finished.")

    # 7. Rolling Window Statistical Features (on a broader but still selected set of features)
    logger.info("Step 7: Generating rolling statistical features...")
    # Select a broader set for rolling stats: key features + original close prices + key Bollinger Band widths
    cols_for_rolling_stats = list(key_features_for_expansion) # Start with key features
    for p in asset_prefixes:
        if f'{p}_Close' in data.columns: cols_for_rolling_stats.append(f'{p}_Close')
        # Construct correct BB_Width column name (e.g. _20_2.0 if std_dev is float)
        # This assumes a common BB window/std_dev is generated and of interest.
        # Example: using the first defined BOLLINGER_WINDOW and BOLLINGER_STD_DEV
        # Ensure BOLLINGER_WINDOWS and BOLLINGER_STD_DEV are not empty before accessing index 0
        if BOLLINGER_WINDOWS and BOLLINGER_STD_DEV:
            bb_width_col_example = f'{p}_BB_Width_{BOLLINGER_WINDOWS[0]}_{BOLLINGER_STD_DEV[0]}'
            if bb_width_col_example in data.columns : cols_for_rolling_stats.append(bb_width_col_example)
        
    cols_for_rolling_stats = list(set(cols_for_rolling_stats)) # Get unique columns
    cols_for_rolling_stats = [col for col in cols_for_rolling_stats if col in data.columns] # Ensure they exist
    logger.debug(f"Generating rolling stats for {len(cols_for_rolling_stats)} columns.")
    data = add_rolling_statistical_features(data, cols_for_rolling_stats, windows=ROLLING_STAT_WINDOWS)
    logger.info("Step 7 finished.")

    # 8. Data Transformations (log, sqrt, square on a broader set of generated numeric features)
    logger.info("Step 8: Generating data transformations...")
    # Select columns for transformation: new numeric features + original asset close/volume
    # Avoid re-transforming already transformed features (e.g., log returns, existing logs/ratios/polynomials).
    transformable_cols = list(numerical_cols_for_transformations) # Start with newly generated numeric features
    for p in asset_prefixes: # Add original close and volume if not already captured
        if f'{p}_Close' in data.columns: transformable_cols.append(f'{p}_Close')
        if f'{p}_Volume' in data.columns: transformable_cols.append(f'{p}_Volume')
        
    # Filter out columns that are likely already transformed or unsuitable
    transformable_cols = [
        c for c in list(set(transformable_cols)) # Unique
        if 'LogReturn' not in c and '_log' not in c and '_ratio_' not in c and '_poly_' not in c and '_sq' not in c and '_sqrt' not in c
        and c in data.columns # Ensure column exists
    ]
    # Limit the number of columns for these transformations to prevent excessive feature explosion
    max_cols_for_std_transforms = 100 
    logger.debug(f"Applying standard transformations to up to {max_cols_for_std_transforms} selected columns from {len(transformable_cols)} candidates.")
    data = add_data_transformations(data, transformable_cols[:max_cols_for_std_transforms])
    logger.info("Step 8 finished.")

    # 9. Lagged Features (apply to a VERY WIDE range of generated features)
    logger.info("Step 9: Generating lagged features...")
    # Lag almost everything generated up to this point, except time features or features that are already lags.
    cols_to_lag = [col for col in data.columns if not col.startswith('time_') and '_lag_' not in col]
    logger.debug(f"Generating lags for {len(cols_to_lag)} columns using {len(LAG_PERIODS)} lag periods.")
    data = add_lagged_features(data, cols_to_lag, lag_periods=LAG_PERIODS)
    logger.info("Step 9 finished.")
    
    # --- Final Processing ---
    logger.info(f"Total features generated before final NaN handling: {len(data.columns)}")

    # Handle missing values (NaNs, Infs) that might have been introduced or remain
    # Replace Inf/-Inf with NaN first.
    # Note: cupy.inf might behave differently with .replace in some cuDF versions.
    # Using masked_assign is generally safer with cuDF/cuPy for Inf handling.
    logger.debug("Replacing Inf values with NaN...")
    for col_name in data.columns:
        col = data[col_name]
        # Check if column is float type, as Inf replacement is relevant for float columns
        if col.dtype in [cp.float32, cp.float64, np.float32, np.float64]:
            # Using masks for inf replacement as it's generally safer with cudf/cupy
            # Check if cupy isinf can be called on the column directly
            try:
                # Ensure we are working with the underlying cupy array for cp.isinf if it's a cuDF series
                values_to_check = col.values if hasattr(col, 'values') else col
                is_inf_mask = cp.isinf(values_to_check)
                if cp.any(is_inf_mask): # Only assign if there are Infs
                    # If col is a cuDF Series, direct assignment with a boolean mask (is_inf_mask) and np.nan should work.
                    # However, masked_assign is more explicit for cuDF if is_inf_mask is a cuDF boolean Series.
                    # Let's ensure is_inf_mask is a cuDF Series if col is.
                    if isinstance(col, cudf.Series):
                        # Create a boolean cuDF Series from the cupy boolean array
                        is_inf_mask_cudf = cudf.Series(is_inf_mask, index=col.index)
                        data[col_name] = col. όχι(is_inf_mask_cudf, np.nan) # cuDF's fillna based on mask
                    else: # if col is a cupy array
                         col[is_inf_mask] = np.nan # Direct assignment for cupy array
                         data[col_name] = col 
            except TypeError: # If col is not directly usable by cp.isinf (e.g. non-numeric mixed type after bad op)
                logger.warning(f"Could not process column {col_name} for Inf replacement due to dtype {col.dtype}. Skipping Inf replacement for this column.")
            except Exception as e: # Catch any other unexpected errors
                 logger.error(f"Unexpected error during Inf replacement for column {col_name}: {e}")


    # Fill remaining NaNs - ffill and bfill first to propagate existing values, then fill remaining with 0.0
    # This is a common strategy but might need adjustment based on specific feature characteristics.
    logger.debug("Applying final ffill, bfill, and fillna(0.0) to handle remaining NaNs...")
    data = data.ffill() # Forward fill
    data = data.bfill() # Backward fill
    data = data.fillna(0.0) # Fill any remaining NaNs with 0.0
    
    logger.info(f"Feature engineering complete. Total features after final NaN handling: {len(data.columns)}")
    
    # Optional: Convert back to pandas if downstream components require it.
    # For this pipeline, returning cuDF is preferred for GPU acceleration.
    # data_pd = data.to_pandas()
    # return data_pd
    return data