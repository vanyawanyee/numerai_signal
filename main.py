import logging
from src.data.data_loader import load_market_data, load_economic_indicators
from src.features.feature_engineering import engineer_features
from src.models.model_training import train_models
from src.models.model_evaluation import evaluate_models, print_evaluation_results
from src.visualization.visualization import plot_results, plot_feature_importance
from src.utils.spark_integration import run_spark_lgbm_model
from src.data.numerai_integration import download_numerai_data, process_numerai_data
from src.utils.environment_setup import main as setup_environment
from config.config import *

logging.basicConfig(level=logging.INFO)

def main():
    # Set up the environment
    spark = setup_environment()
    
    # Load data
    market_data = load_market_data(start_date=config.START_DATE, end_date=config.END_DATE)
    economic_indicators = load_economic_indicators(api_key=config.FRED_API_KEY)
    
    # Engineer features
    features = engineer_features(market_data, economic_indicators)
    
    # Train models
    models, scalers = train_models(features)
    
    # Evaluate models
    results = evaluate_models(models, features, scalers)
    print_evaluation_results(results)
    
    # Visualize results
    plot_results(results)
    plot_feature_importance(models['lgbm_dax'], features, "DAX")
    plot_feature_importance(models['lgbm_tesla'], features, "Tesla")
    
    # Run Spark LightGBM model
    spark_auc = run_spark_lgbm_model()
    logging.info(f"Spark LightGBM Model AUC: {spark_auc}")
    
    # Download and process Numerai data
    numerai_data = download_numerai_data()
    processed_numerai_data = process_numerai_data(numerai_data)
    
    logging.info("Analysis complete.")
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()