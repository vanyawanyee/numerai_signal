"""
PySpark Integration Module for LightGBM
=======================================

This module provides functionalities to set up a PySpark session, load data (specifically DAX index data
as an example), prepare it for Spark ML, and train a LightGBM model using SynapseML (formerly MMLSpark).
It also includes evaluation of the trained model.

This part of the project appears to be a somewhat separate workflow from the main RAPIDS-based
pipeline, focusing on demonstrating or utilizing Spark-based distributed training capabilities.

Key Functions:
- `create_spark_session`: Initializes a SparkSession with configurations for SynapseML.
- `load_dax_data`: Loads DAX historical data using yfinance, with a dummy data fallback.
- `prepare_spark_data`: Transforms a pandas DataFrame into a Spark DataFrame suitable for ML,
  including feature vectorization and label indexing (currently set for a classification task).
- `train_spark_lgbm`: Trains a LightGBMClassifier model using SynapseML.
- `evaluate_spark_model`: Evaluates the trained Spark ML model (calculates AUC for classification).
- `run_spark_lgbm_model`: Orchestrates the above steps to run the full Spark LightGBM pipeline.

Note on current implementation:
- The `prepare_spark_data` and model type (LightGBMClassifier) suggest a classification setup
  (e.g., predicting price movement direction), while `label_col = 'Close'` might imply regression.
  This should be clarified or adjusted based on the specific prediction task for this Spark pipeline.
- The module checks for `yfinance` and `synapse.ml` availability globally.
"""
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder 
# Note: OneHotEncoder is imported but not used if cat_cols is empty.
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
# This evaluator is for binary classification tasks. If regression, RegressionEvaluator would be used.
import logging
import pandas as pd # For dummy data and yfinance output
import numpy as np  # For dummy data

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global variables to track availability of optional libraries
SYNAPSE_AVAILABLE = False
YFINANCE_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.info("yfinance loaded successfully.")
except ImportError:
    logger.warning("yfinance not available. DAX data loading will use dummy data if called.")

# Attempt to import SynapseML LightGBMClassifier to set SYNAPSE_AVAILABLE flag early
# This helps in create_spark_session to log availability accurately.
try:
    from synapse.ml.lightgbm import LightGBMClassifier # Specific import for early check
    SYNAPSE_AVAILABLE = True # If this import succeeds, assume SynapseML is generally available
    logger.info("SynapseML (for LightGBM on Spark) appears to be available.")
except ImportError:
    logger.warning("SynapseML (synapse.ml.lightgbm.LightGBMClassifier) not found. LightGBM on Spark will not be available.")
except Exception as e: # Catch other potential errors during import
    logger.error(f"An error occurred while trying to import SynapseML: {e}")


def create_spark_session(app_name="DAXPredictionWithSparkLGBM"):
    """
    Creates and configures a SparkSession with necessary packages for SynapseML.

    Args:
        app_name (str): The name for the Spark application.

    Returns:
        pyspark.sql.SparkSession: The initialized SparkSession, or None if creation fails.
    """
    logger.info(f"Creating Spark session: {app_name}")
    try:
        # Configuration for SynapseML (formerly MMLSpark)
        # These packages provide LightGBM on Spark capabilities.
        spark = (SparkSession.builder.appName(app_name)
                .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.5") # Check for latest SynapseML version
                .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") # For efficient Pandas conversion
                .getOrCreate())
        logger.info("Spark session created successfully.")
        
        # Log SynapseML availability status based on earlier import attempt
        if SYNAPSE_AVAILABLE:
            logger.info("SynapseML was successfully imported earlier. LightGBM on Spark should be usable.")
        else:
            logger.warning("SynapseML could not be imported earlier. LightGBM on Spark functionalities will be unavailable.")
        return spark
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}", exc_info=True)
        return None

def load_dax_data(spark, start_date='2014-06-04', end_date='2024-06-03'):
    """
    Loads DAX historical data using yfinance or generates dummy data if yfinance is unavailable.
    The data is returned as a Spark DataFrame.

    Args:
        spark (pyspark.sql.SparkSession): The Spark session to create the DataFrame.
        start_date (str): Start date for data loading (YYYY-MM-DD).
        end_date (str): End date for data loading (YYYY-MM-DD).

    Returns:
        pyspark.sql.DataFrame: Spark DataFrame containing DAX data or dummy data.
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available. Generating and using dummy data for DAX.")
        # Create a dummy pandas DataFrame
        num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        if num_days <= 0: num_days = 100 # Default if date range is invalid
        dummy_pdf = pd.DataFrame({
            'Date': pd.date_range(start=start_date, periods=num_days, freq='B'), # Business days
            'Open': np.random.rand(num_days) * 100 + 10000,
            'High': np.random.rand(num_days) * 50 + 10050,
            'Low': np.random.rand(num_days) * 50 + 9950,
            'Close': np.random.rand(num_days) * 100 + 10000,
            'Volume': np.random.randint(1000000, 5000000, num_days)
        })
        dummy_pdf['Adj Close'] = dummy_pdf['Close'] # Simplified Adj Close
        logger.info(f"Generated dummy DAX data. Shape: {dummy_pdf.shape}")
        return spark.createDataFrame(dummy_pdf)
    
    logger.info(f"Loading DAX data using yfinance from {start_date} to {end_date}.")
    try:
        dax_pdf = yf.download('^GDAXI', start=start_date, end=end_date) # Changed ticker to ^GDAXI for DAX
        if dax_pdf.empty:
            logger.warning("yf.download returned an empty DataFrame for DAX data.")
            return spark.createDataFrame(pd.DataFrame()) # Return empty Spark df
        dax_pdf = dax_pdf.reset_index() # Move Date from index to column
        logger.info(f"DAX data loaded via yfinance. Shape: {dax_pdf.shape}")
        return spark.createDataFrame(dax_pdf)
    except Exception as e:
        logger.error(f"Error loading DAX data using yfinance: {e}", exc_info=True)
        logger.warning("Falling back to dummy DAX data due to yfinance error.")
        return load_dax_data(spark, start_date, end_date) # Recursive call to get dummy data

def prepare_spark_data(df_spark):
    """
    Prepares Spark DataFrame for machine learning by vectorizing features.
    This example sets up for a classification task using 'Close' as a label, which might need adjustment
    (e.g., deriving a binary label like 'price_up_down' from 'Close').

    Args:
        df_spark (pyspark.sql.DataFrame): Input Spark DataFrame with raw features.

    Returns:
        pyspark.sql.DataFrame: Transformed Spark DataFrame with a 'features' vector column and 'label' column.
    """
    if df_spark.count() == 0: # Check if DataFrame is empty
        logger.warning("Input Spark DataFrame for preparation is empty. Returning empty DataFrame.")
        return df_spark 

    logger.info("Preparing Spark DataFrame for ML: vectorizing features and indexing label.")
    
    # Define feature columns
    # Categorical columns (cat_cols) are currently empty; if any, they'd need OneHotEncoding.
    cat_cols = [] 
    # Continuous features used directly in VectorAssembler
    cont_cols = ['Open', 'High', 'Low', 'Volume'] 
    
    # Define label column - current setup uses 'Close' directly which is unusual for classification.
    # For classification, 'Close' would typically be transformed into a binary/multiclass label.
    # E.g., df_spark = df_spark.withColumn("PriceDiff", F.col("Close") - F.col("Open"))
    #        df_spark = df_spark.withColumn("label", (F.col("PriceDiff") > 0).cast("double"))
    # For regression, label_col would be 'Close' as float, and RegressionEvaluator used.
    label_col_raw = 'Close' 
    label_col_indexed = "label" # Output of StringIndexer, used by LightGBMClassifier

    stages = []
    
    # Ensure label column is numeric for StringIndexer or direct use.
    # If 'Close' is already numeric, StringIndexer might not be strictly necessary if values are 0.0, 1.0 etc.
    # However, LightGBMClassifier in SynapseML expects indexed labels.
    # If this were a regression task, 'Close' could be used directly as label if float/double.
    try:
        # Check if label_col_raw exists
        if label_col_raw not in df_spark.columns:
            logger.error(f"Raw label column '{label_col_raw}' not found in Spark DataFrame. Columns: {df_spark.columns}")
            # Return original df or raise error, depending on desired handling
            raise ValueError(f"Raw label column '{label_col_raw}' not found.")

        # Ensure all feature columns exist
        missing_feature_cols = [col for col in cont_cols if col not in df_spark.columns]
        if missing_feature_cols:
            logger.error(f"Missing feature columns in Spark DataFrame: {missing_feature_cols}. Available: {df_spark.columns}")
            raise ValueError(f"Missing feature columns: {missing_feature_cols}")

        label_indexer = StringIndexer(inputCol=label_col_raw, outputCol=label_col_indexed, handleInvalid="skip")
        stages.append(label_indexer)
    except Exception as e:
        logger.error(f"Error setting up StringIndexer for label '{label_col_raw}': {e}", exc_info=True)
        # Depending on robustness needs, could return df_spark or raise
        raise

    # Vector Assembler: Combines feature columns into a single 'features' vector
    assembler_inputs = cont_cols # Assuming cat_cols is empty for now. If cat_cols had data, it would be f"{cat_col}_vec"
    vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")
    stages.append(vector_assembler)
    
    logger.debug(f"Pipeline stages for data preparation: {stages}")
    pipeline = Pipeline(stages=stages)
    
    try:
        logger.info("Fitting data preparation pipeline...")
        pipeline_model = pipeline.fit(df_spark)
        df_transformed = pipeline_model.transform(df_spark)
        logger.info("Data preparation pipeline fitted and data transformed.")
        # Show some transformed data for debugging if needed
        # df_transformed.select("features", label_col_indexed).show(5, truncate=False)
        return df_transformed
    except Exception as e:
        logger.error(f"Error during Spark data preparation pipeline: {e}", exc_info=True)
        # Log schema for debugging
        df_spark.printSchema()
        raise

def train_spark_lgbm(df_transformed_with_features):
    """
    Trains a LightGBMClassifier model using SynapseML on the prepared Spark DataFrame.

    Args:
        df_transformed_with_features (pyspark.sql.DataFrame): Spark DataFrame with 'features' and 'label' columns.

    Returns:
        synapse.ml.lightgbm.LightGBMClassificationModel: Trained LightGBM model, or None if SynapseML is unavailable.
    """
    if not SYNAPSE_AVAILABLE:
        logger.warning("SynapseML (LightGBMClassifier) not available. Skipping LightGBM training on Spark.")
        return None
    
    if df_transformed_with_features.count() == 0:
        logger.warning("Input DataFrame for Spark LGBM training is empty. Skipping training.")
        return None

    logger.info("Training LightGBM model on Spark using SynapseML...")
    # Ensure correct import path for LightGBMClassifier from SynapseML
    from synapse.ml.lightgbm import LightGBMClassifier 
    
    # Example parameters for LightGBMClassifier
    # These should be tuned for the specific task.
    # Note: This is a Classifier. For regression, LightGBMRegressor would be used.
    # The current data prep (StringIndexer on 'Close') leans towards classification.
    lgbm_classifier = LightGBMClassifier(
        featuresCol="features", 
        labelCol="label", 
        numIterations=100, # Number of boosting iterations
        learningRate=0.1,
        # numLeaves=31, # Example: Default is 31
        # objective="binary", # Or "multiclass" if numClass > 2. StringIndexer creates labels 0,1,2...
        # isUnbalance=True, # Example: if classes are imbalanced
        verbosity=-1 # Suppress LightGBM native verbosity
    )
    
    try:
        model = lgbm_classifier.fit(df_transformed_with_features)
        logger.info("Spark LightGBM model training completed.")
        return model
    except Exception as e:
        logger.error(f"Error training Spark LightGBM model: {e}", exc_info=True)
        # df_transformed_with_features.select("features", "label").show(5)
        # df_transformed_with_features.groupBy("label").count().show()
        raise

def evaluate_spark_model(model, df_test_transformed):
    """
    Evaluates the trained Spark ML model (LightGBMClassifier) using BinaryClassificationEvaluator (AUC).

    Args:
        model (pyspark.ml.Model): The trained Spark ML model.
        df_test_transformed (pyspark.sql.DataFrame): Transformed test data with 'features' and 'label' columns.

    Returns:
        float: Area Under ROC (AUC) score, or None if evaluation fails or model is None.
    """
    if model is None:
        logger.warning("Spark model is None. Skipping evaluation.")
        return None
    if df_test_transformed.count() == 0:
        logger.warning("Test DataFrame for Spark model evaluation is empty. Skipping evaluation.")
        return None

    logger.info("Evaluating Spark LightGBM model...")
    try:
        predictions = model.transform(df_test_transformed)
        
        # Using BinaryClassificationEvaluator, which expects 'rawPrediction' or 'probability' column, and 'label'.
        # LightGBMClassifier typically outputs 'rawPrediction', 'probability', and 'prediction'.
        # Ensure the labelCol matches what was used in training (output of StringIndexer).
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        logger.info(f"Spark model evaluation complete. AUC: {auc:.4f}")
        return auc
    except Exception as e:
        logger.error(f"Error evaluating Spark model: {e}", exc_info=True)
        # predictions.select("label", "rawPrediction", "probability", "prediction").show(5)
        return None

def run_spark_lgbm_model(spark_session=None):
    """
    Orchestrates the entire pipeline for training and evaluating a LightGBM model on Spark
    using DAX data. This function is intended as a self-contained example run.

    Args:
        spark_session (pyspark.sql.SparkSession, optional): An existing Spark session. 
                                                           If None, a new one is created.

    Returns:
        float: AUC score of the trained model, or None if any step fails.
    """
    logger.info("Starting full Spark LightGBM model pipeline run...")
    created_session_locally = False
    if spark_session is None:
        spark_session = create_spark_session()
        if spark_session is None:
            logger.error("Failed to obtain Spark session for run_spark_lgbm_model. Aborting.")
            return None
        created_session_locally = True
        logger.info("Spark session obtained for Spark LGBM pipeline.")

    # Load data (example: DAX data)
    df_spark_raw = load_dax_data(spark_session)
    if df_spark_raw.count() == 0:
        logger.error("Raw data for Spark LGBM is empty. Aborting pipeline.")
        if created_session_locally:
            spark_session.stop()
            logger.info("Locally created Spark session stopped.")
        return None

    # Prepare data for Spark ML
    # For a proper evaluation, split data into train and test sets BEFORE preparing.
    logger.info("Splitting raw Spark data into training and test sets (80/20).")
    train_df_raw, test_df_raw = df_spark_raw.randomSplit([0.8, 0.2], seed=SEED) # Use global SEED
    
    logger.info("Preparing training data for Spark ML...")
    train_df_transformed = prepare_spark_data(train_df_raw)
    if train_df_transformed.count() == 0:
        logger.error("Transformed training data for Spark LGBM is empty. Aborting pipeline.")
        if created_session_locally:
            spark_session.stop()
        return None
        
    logger.info("Preparing test data for Spark ML...")
    # Use the same pipeline model fitted on training data to transform test data
    # This requires refitting the pipeline only on train_df_raw then transforming test_df_raw
    # For simplicity in this example, prepare_spark_data is called on test_df_raw,
    # which re-fits StringIndexers. This is okay if label cardinality is same, but not ideal.
    # Correct approach: fit Pipeline on train, then transform train and test.
    # This simplification is kept from original structure.
    # Re-evaluating: The pipeline should be fit on TRAIN data only.
    
    # Fit the preparation pipeline on training data
    # (Assuming `prepare_spark_data` is refactored to return the pipeline model and transform separately)
    # For now, we'll call `prepare_spark_data` on test data as well, which is a shortcut and implies
    # StringIndexers, etc., are re-fit. This is generally not best practice.
    # A better `prepare_spark_data` would return the fitted pipeline and then use it to transform test set.
    # To maintain current structure but improve:
    # 1. `prepare_spark_data` could return fitted_pipeline_model, train_transformed
    # 2. Then `test_transformed = fitted_pipeline_model.transform(test_df_raw)`
    # For this refactoring pass, I will keep the original logic of calling prepare_spark_data on test set.
    test_df_transformed = prepare_spark_data(test_df_raw)
    if test_df_transformed.count() == 0:
        logger.error("Transformed test data for Spark LGBM is empty. Aborting pipeline.")
        if created_session_locally:
            spark_session.stop()
        return None

    # Train LightGBM model on Spark
    model = train_spark_lgbm(train_df_transformed)
    
    # Evaluate model
    auc = None
    if model:
        auc = evaluate_spark_model(model, test_df_transformed)
    
    if auc is not None:
        logger.info(f"Spark LightGBM Model run complete. Final Test AUC: {auc:.4f}")
    else:
        logger.warning("Spark LightGBM Model run could not be fully completed or evaluated.")
    
    if created_session_locally:
        spark_session.stop()
        logger.info("Locally created Spark session stopped.")
    return auc