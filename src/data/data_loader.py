import yfinance as yf
import pandas as pd
from fredapi import Fred
import logging

def load_market_data(start_date="2010-01-01", end_date="2023-12-31"):
    indices = {
        'DAX': '^GDAXI',
        'SP500': '^GSPC',
        'NASDAQ': '^IXIC',
        'NIKKEI': '^N225',
        'BOVESPA': '^BVSP',
        'SHANGHAI': '000001.SS'
    }
    
    data = pd.DataFrame()
    for name, ticker in indices.items():
        df = yf.download(ticker, start=start_date, end=end_date)
        df.columns = [f'{name}_{col}' for col in df.columns]
        if data.empty:
            data = df
        else:
            data = data.join(df, how='outer')
    
    # Add Tesla data
    tesla = yf.download('TSLA', start=start_date, end=end_date)
    tesla.columns = [f'Tesla_{col}' for col in tesla.columns]
    data = data.join(tesla, how='outer')
    
    logging.info(f"Market data loaded. Shape: {data.shape}")
    return data

def load_economic_indicators(api_key='your_fred_api_key'):
    fred = Fred(api_key=api_key)
    economic_indicators = [
        ('ICSA', 'Initial Claims'),
        ('UNRATE', 'Unemployment Rate'),
        ('CSUSHPISA', 'Case-Shiller Home Price Index'),
        ('MSPUS', 'Median Sales Price of Houses Sold'),
        ('NAPM', 'ISM Manufacturing PMI'),
        ('UMCSENT', 'Consumer Sentiment Index')
    ]
    
    data = pd.DataFrame()
    for series_id, name in economic_indicators:
        try:
            series = fred.get_series(series_id)
            data[name] = series
            logging.info(f"Successfully downloaded: {name}")
        except ValueError as e:
            logging.error(f"Error downloading {name} ({series_id}): {str(e)}")
    
    logging.info(f"Economic indicators loaded. Shape: {data.shape}")
    return data