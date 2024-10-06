from pyspark.sql import Window
from pyspark.sql.functions import lag, col, stddev, skewness, kurtosis
from config import FeatureCFG
import pyspark.sql.functions as F

def add_feats(df, group_col='ticker', sort_col='date'):
    window = Window.partitionBy(group_col).orderBy(sort_col)
    for feature in FeatureCFG.BASIC_FEATURES:
        for alpha in FeatureCFG.EMA_ALPHAS:
            df = df.withColumn(f'EMA_{feature}_{alpha}', F.avg(df[feature]).over(window.rowsBetween(-int(1/(1-alpha)), 0)))
        for window_size in FeatureCFG.MEAN_WINDOWS:
            df = df.withColumn(f'MEAN_{feature}_{window_size}', F.avg(df[feature]).over(window.rowsBetween(-window_size, 0)))
        if FeatureCFG.RETURN_FEATURE:
            df = df.withColumn(f'{feature}_Return', (df[feature] - lag(df[feature]).over(window)) / lag(df[feature]).over(window))
        if FeatureCFG.VOLUME_CHANGE_FEATURE:
            df = df.withColumn(f'{feature}_Change', (df[feature] - lag(df[feature]).over(window)) / lag(df[feature]).over(window))
        if FeatureCFG.STD_FEATURE:
            df = df.withColumn(f'STD_{feature}', stddev(df[feature]).over(window))
        if FeatureCFG.SKEW_FEATURE:
            df = df.withColumn(f'SKEW_{feature}', skewness(df[feature]).over(window))
        if FeatureCFG.KURTOSIS_FEATURE:
            df = df.withColumn(f'KURTOSIS_{feature}', kurtosis(df[feature]).over(window))
    return df

def add_advanced_feats(df, group_col='ticker', sort_col='date'):
    window = Window.partitionBy(group_col).orderBy(sort_col)
    df = df.withColumn('financial_volume', df['Close'] * df['Volume'])
    df = df.withColumn('financial_volume_change', (df['financial_volume'] - lag(df['financial_volume']).over(window)) / lag(df['financial_volume']).over(window))
    for feature in ['Open', 'High', 'Low', 'Close']:
        df = df.withColumn(f'{feature}_return', (df[feature] - lag(df[feature]).over(window)) / lag(df[feature]).over(window))
    df = df.withColumn('open_close_range', (df['Open'] - df['Close']).abs())
    df = df.withColumn('high_low_range', (df['High'] - df['Low']).abs())
    return df
