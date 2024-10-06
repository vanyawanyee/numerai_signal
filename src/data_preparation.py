from src.initialize import initialization
from config.config import  ROOT_DIR, config_dict
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, to_date
from pyspark.sql.types import StringType
from pyspark.sql import Row
from datetime import datetime, timedelta

spark, napi = initialization()
full_run = config_dict['RUN_PARAMETER']['full_run']

# spark = SparkSession.builder.appName("numerai").getOrCreate()

def load_ticker(ticker_file_name:str):
    ticker_path = ROOT_DIR.joinpath('data/'+ticker_file_name)
    df_ticker = spark.read.csv(str(ticker_path), header=True, sep=',')
    ticker_col_names = {'ticker_bb':'bloomberg_ticker', 'yahoo':'yahoo_ticker'}
    for old_col, new_col in ticker_col_names.items():
        df_ticker = df_ticker.withColumnRenamed(old_col, new_col)

    # df = df.select(
    # [df[col].alias(column_mapping[col]) if col in column_mapping else df[col] for col in df.columns]
    # )
    return df_ticker


def load_train_validation_data(mode='local'):
    train_file = config_dict['DATA']['train_file']
    validation_file = config_dict['DATA']['validation_file']
    if mode == 'update':
        napi.download_dataset(train_file)
        napi.download_dataset(validation_file)


    df_train_path = ROOT_DIR.joinpath(train_file)
    df_validation_path = ROOT_DIR.joinpath(validation_file)
    df_train = spark.read.parquet(str(df_train_path))
    df_validation = spark.read.parquet(str(df_validation_path))

    # add this for development - testing purpose using smaller dataframe
    df_train=df_train.limit(100)
    df_validation=df_validation.limit(100)

    df_train = df_train.withColumn('dataset_type', lit('training'))
    df_validation = df_validation.withColumn('dataset_type', lit('validation'))
    targets = df_train.union(df_validation)

    # Convert 'date' column to date format
    targets = targets.withColumn('date', to_date(col('date').cast(StringType()), 'yyyy-MM-dd'))

    return targets


def load_yahoo_price_data(mode='local'):
    if mode == 'update':
        # TODO: add the code to download the data online
        pass
    stock_price_path = ROOT_DIR.joinpath(config_dict['DATA']['yahoo_price_file'])
    df_stock_price = spark.read.parquet(str(stock_price_path))

    df_stock_price = df_stock_price.limit(1000)
    df_yahoo_data = df_stock_price.withColumn('date', to_date(col('date').cast(StringType()), 'yyyyMMdd'))
    return df_yahoo_data


def load_sector_data(mode='local'):
    if mode == 'update':
        # TODO: add online update
        pass
    sector_path = ROOT_DIR.joinpath(config_dict['DATA']['sector_file'])
    df_sector = spark.read.csv(str(sector_path),inferSchema="true", header="true")
    df_sector = df_sector.select('SEDOL','ticker_bb','GICS_subindustry').withColumnRenamed('ticker_bb','bloomberg_ticker')

    return df_sector


def combine_sector_and_yahoo_price_data(df_yahoo_data, df_sector):
    sector_close_volume = df_yahoo_data.join(
        df_sector, df_yahoo_data.ticker == df_sector.bloomberg_ticker, "left"
    )
    return sector_close_volume


def get_friday_date(start_date, end_date=datetime.today().date()):
    """
    this function returns a list of string of dates that fall on Friday
    params:
    start_iter_date -> given in date/str format e.g. date(yyyy,MM,dd) | 'YYYY-mm-dd'
    end_date -> default value is today

    example
    get_friday_date(date(2020,2,28))
    """
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    friday_date = []
    while start_date <= end_date:
        if start_date.weekday() == 4:
            friday_date.append(start_date.strftime('%Y-%m-%d'))
            start_date = start_date + timedelta(weeks=1)
        else:
            start_date = start_date + timedelta(days=1)

    return friday_date


def create_pyspark_dataframe_from_list(data_list:list,column_name:str):
    rdd_from_list = spark.sparkContext.parallelize(data_list)
    row_element = rdd_from_list.map(lambda x: Row(x))
    df_from_list = spark.createDataFrame(row_element, [column_name])

    return df_from_list


def create_friday_date_dataframe():
    friday_start_date = config_dict['RUN_PARAMETER']['friday_start_date']
    df_friday_date = create_pyspark_dataframe_from_list(get_friday_date(friday_start_date), 'friday_date')
    return df_friday_date



def load_input_data():
    df_yahoo_data = load_yahoo_price_data()
    df_sector_data = load_sector_data()
    df_train_validation_full_data = load_train_validation_data()
    df_friday_date = create_friday_date_dataframe()

    return df_yahoo_data, df_sector_data, df_train_validation_full_data, df_friday_date


def process_data(df_yahoo_data, df_sector_data, df_train_validation_full_data, df_friday_date):

    df_yahoo_sector_data = combine_sector_and_yahoo_price_data(df_yahoo_data, df_sector_data)

    # keep only records on friday
    df_yahoo_friday_price = (
        df_yahoo_sector_data
        .join(df_friday_date, [df_yahoo_sector_data.date == df_friday_date.friday_date], how='inner'
        )
        #.withColumnRenamed('friday_date', 'fri_date')
    )

    df_full_data = (
        df_train_validation_full_data
        .alias('numerai')
        .join(df_yahoo_friday_price.alias('yahoo').drop(col('yahoo.date')),
              (col('yahoo.ticker') == col('numerai.numerai_ticker')) &
              (col('yahoo.friday_date') == col('numerai.date')),
              how='right'
              )
        .drop(col('numerai.numerai_ticker'))
    )

    df_full_data_subset = (
        df_full_data
        .select(
            col('ticker'),
            col('SEDOL'),
            col('date'),
            col('data_type'),
            col('Close'),
            col('High'),
            col('Low'),
            col('Volume'),
            col('GICS_subindustry'),
            col('friday_date'),
            col('target')
        )
    )

    return df_full_data_subset