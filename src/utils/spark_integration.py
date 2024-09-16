import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Global variables to track availability
SYNAPSE_AVAILABLE = False
YFINANCE_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    print("yfinance not available. DAX data loading will not be available.")

def create_spark_session():
    global SYNAPSE_AVAILABLE
    spark = (SparkSession.builder.appName("DAXPrediction")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .getOrCreate())
    
    try:
        import synapse.ml
        from synapse.ml.lightgbm import LightGBMClassifier
        SYNAPSE_AVAILABLE = True
    except ImportError:
        print("Synapse ML not available. LightGBM on Spark will not be available.")
    
    return spark

def load_dax_data(start_date='2014-06-04', end_date='2024-06-03'):
    if not YFINANCE_AVAILABLE:
        print("yfinance not available. Using dummy data.")
        return pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date),
                             'Open': np.random.rand(100),
                             'High': np.random.rand(100),
                             'Low': np.random.rand(100),
                             'Close': np.random.rand(100),
                             'Volume': np.random.randint(1000, 10000, 100)})
    
    return yf.download('DAX', start=start_date, end=end_date).reset_index()

def prepare_spark_data(spark, df_pandas):
    df = spark.createDataFrame(df_pandas)
    
    cat_cols = []
    cont_cols = ['Open', 'High', 'Low', 'Volume']
    label_col = 'Close'
    
    stages = []
    
    # Label indexer
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label")
    stages += [label_indexer]
    
    # Categorical columns encoding
    for cat_col in cat_cols:
        string_indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_index")
        encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[f"{cat_col}_vec"])
        stages += [string_indexer, encoder]
    
    # Vector assembler
    assembler_inputs = [f"{cat_col}_vec" for cat_col in cat_cols] + cont_cols
    vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    stages += [vector_assembler]
    
    # Create pipeline
    pipeline = Pipeline(stages=stages)
    
    # Fit pipeline
    pipeline_model = pipeline.fit(df)
    
    # Transform data
    df_transformed = pipeline_model.transform(df)
    
    return df_transformed

def train_spark_lgbm(df_transformed):
    if not SYNAPSE_AVAILABLE:
        print("Synapse ML not available. Skipping LightGBM training.")
        return None
    
    from synapse.ml.lightgbm import LightGBMClassifier
    classifier = LightGBMClassifier(featuresCol="features", labelCol="label", numIterations=100)
    model = classifier.fit(df_transformed)
    return model

def evaluate_spark_model(model, df_transformed):
    if model is None:
        print("Model is None. Skipping evaluation.")
        return None
    
    predictions = model.transform(df_transformed)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
    auc = evaluator.evaluate(predictions)
    return auc

def run_spark_lgbm_model():
    spark = create_spark_session()
    df_pandas = load_dax_data()
    df_transformed = prepare_spark_data(spark, df_pandas)
    model = train_spark_lgbm(df_transformed)
    auc = evaluate_spark_model(model, df_transformed)
    
    if auc is not None:
        print(f"Spark LightGBM Model AUC: {auc}")
    else:
        print("Spark LightGBM Model could not be evaluated.")
    
    spark.stop()
    return auc