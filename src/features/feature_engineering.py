import pandas as pd
import numpy as np

def EWMA(series, n):
    return series.ewm(span=n, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def calculate_roc(prices, period=12):
    return prices.pct_change(periods=period)

def add_indicators(df, prefix):
    df[f'{prefix}_EMA_5'] = EWMA(df[f'{prefix}_Close'], 5)
    df[f'{prefix}_EMA_20'] = EWMA(df[f'{prefix}_Close'], 20)
    df[f'{prefix}_SMA_5'] = df[f'{prefix}_Close'].rolling(window=5).mean()
    df[f'{prefix}_SMA_20'] = df[f'{prefix}_Close'].rolling(window=20).mean()
    df[f'{prefix}_RSI'] = calculate_rsi(df[f'{prefix}_Close'])
    df[f'{prefix}_MACD'] = calculate_macd(df[f'{prefix}_Close'])
    df[f'{prefix}_BB_Upper'], df[f'{prefix}_BB_Lower'] = calculate_bollinger_bands(df[f'{prefix}_Close'])
    df[f'{prefix}_ROC'] = calculate_roc(df[f'{prefix}_Close'])
    df[f'{prefix}_Volume_Change'] = df[f'{prefix}_Volume'].pct_change()
    df[f'{prefix}_Volume_MA5'] = df[f'{prefix}_Volume'].rolling(window=5).mean()

def add_lagged_features(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

def engineer_features(market_data, economic_indicators):
    data = market_data.join(economic_indicators, how='outer')
    
    # Add random indicator
    np.random.seed(42)
    data['Random_Indicator'] = np.random.randn(len(data))
    
    # Calculate log returns
    for prefix in ['DAX', 'SP500', 'NASDAQ', 'NIKKEI', 'BOVESPA', 'SHANGHAI', 'Tesla']:
        data[f'{prefix}_LogReturn'] = np.log(data[f'{prefix}_Close'] / data[f'{prefix}_Close'].shift(1))
        add_indicators(data, prefix)
    
    # Add lagged features
    lagged_columns = [f'{prefix}_LogReturn' for prefix in ['DAX', 'SP500', 'NASDAQ', 'NIKKEI', 'BOVESPA', 'SHANGHAI', 'Tesla']] + list(economic_indicators.columns)
    add_lagged_features(data, lagged_columns, [1, 2, 3])
    
    # Handle missing values
    data = data.ffill().bfill()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    return data