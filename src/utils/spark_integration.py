from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import yfinance as yf

try:
    from synapse.ml.lightgbm import LightGBMClassifier
    SYNAPSE_AVAILABLE = True
except ImportError:
    print("Synapse ML not available. LightGBM on Spark will not be available.")
    SYNAPSE_AVAILABLE = False

def create_spark_session():
    return (SparkSession.builder.appName("DAXPrediction")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .getOrCreate())

def load_dax_data(start_date='2014-06-04', end_date='2024-06-03'):
    df_pandas = yf.download('DAX', start=start_date, end=end_date)
    return df_pandas.reset_index()

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
    classifier = LightGBMClassifier(featuresCol="features", labelCol="label", numIterations=100)
    model = classifier.fit(df_transformed)
    return model

def evaluate_spark_model(model, df_transformed):
    predictions = model.transform(df_transformed)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
    auc = evaluator.evaluate(predictions)
    return auc

def run_spark_lgbm_model():
    if not SYNAPSE_AVAILABLE:
        print("Synapse ML is not available. Skipping Spark LightGBM model.")
        return None

    spark = create_spark_session()
    df_pandas = load_dax_data()
    df_transformed = prepare_spark_data(spark, df_pandas)
    model = train_spark_lgbm(df_transformed)
    auc = evaluate_spark_model(model, df_transformed)
    
    print(f"Spark LightGBM Model AUC: {auc}")
    
    spark.stop()
    return auc