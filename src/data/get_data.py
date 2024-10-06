import os
import yfinance
from pyspark.sql import SparkSession
import urllib
from pyspark.sql.functions import col
from config import CFG
import numerapi

def fetch_yfinance(ticker_map, start='2002-12-01'):
    numerai_tickers = ticker_map['numerai_ticker']
    yfinance_tickers = ticker_map['yahoo']
    raw_data = yfinance.download(yfinance_tickers.str.cat(sep=' '), start=start, threads=True)
    cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    full_data = raw_data[cols].stack().reset_index()
    full_data.columns = ['date', 'ticker', 'close', 'raw_close', 'high', 'low', 'open', 'volume']
    full_data['ticker'] = full_data.ticker.map(dict(zip(yfinance_tickers, numerai_tickers)))
    return full_data

def get_data():
    spark = SparkSession.builder.getOrCreate()
    current_ds = CFG.CURRENT_DS_OVERRIDE
    ticker_map_pyspark = spark.read.csv(os.path.join(CFG.INPUT_DIR, "ticker_map_w_bbg_enhanced.csv"), header=True, sep=';')
    et_pyspark = spark.createDataFrame(numerapi.NumerAPI().ticker_universe(), ['ticker']).filter(col('ticker')!='bloomberg_ticker')
    ticker_map_pyspark = ticker_map_pyspark.join(et_pyspark, ticker_map_pyspark.bloomberg_ticker == et_pyspark.ticker, how='inner')
    if CFG.FETCH_VIA_API:
        df = fetch_yfinance(ticker_map_pyspark, start='2002-12-01')
    else:
        df = spark.read.parquet(os.path.join(CFG.INPUT_DIR, 'full_data.parquet'))
    return df
