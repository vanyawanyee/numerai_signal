import pytest
from src.data.data_loader import load_market_data, load_economic_indicators
from src.features.feature_engineering import engineer_features
from src.models.model_training import train_models
from src.models.model_evaluation import evaluate_models
from src.utils.spark_integration import run_spark_lgbm_model
from src.data.numerai_integration import download_numerai_data, process_numerai_data
from config.config import START_DATE, END_DATE, FRED_API_KEY

@pytest.fixture
def sample_data():
    market_data = load_market_data(start_date=START_DATE, end_date=END_DATE)
    economic_indicators = load_economic_indicators(api_key=FRED_API_KEY)
    return market_data, economic_indicators

def test_full_pipeline(sample_data):
    market_data, economic_indicators = sample_data
    
    # Test feature engineering
    features = engineer_features(market_data, economic_indicators)
    assert not features.empty, "Engineered features should not be empty"
    
    # Test model training
    models, scalers = train_models(features)
    assert len(models) == 6, "Should have 6 models (LGBM, NN, H2O for both DAX and Tesla)"
    
    # Test model evaluation
    results = evaluate_models(models, features, scalers)
    assert 'dax' in results and 'tesla' in results, "Should have results for both DAX and Tesla"
    
    # Test Spark integration
    spark_auc = run_spark_lgbm_model()
    assert isinstance(spark_auc, float), "Spark AUC should be a float"
    
    # Test Numerai integration
    numerai_data = download_numerai_data()
    processed_numerai_data = process_numerai_data(numerai_data)
    assert not processed_numerai_data.empty, "Processed Numerai data should not be empty"

def test_model_performance(sample_data):
    market_data, economic_indicators = sample_data
    features = engineer_features(market_data, economic_indicators)
    models, scalers = train_models(features)
    results = evaluate_models(models, features, scalers)
    
    for market in ['dax', 'tesla']:
        assert results[market]['ensemble_r2'] > 0, f"Ensemble R2 for {market} should be positive"
        assert results[market]['ensemble_rmse'] < results[market]['lgbm_rmse'], f"Ensemble RMSE for {market} should be lower than LGBM RMSE"

@pytest.mark.skip(reason="This test may take a long time to run")
def test_h2o_automl_performance(sample_data):
    market_data, economic_indicators = sample_data
    features = engineer_features(market_data, economic_indicators)
    models, scalers = train_models(features)
    results = evaluate_models(models, features, scalers)
    
    for market in ['dax', 'tesla']:
        assert results[market]['h2o_r2'] > 0.1, f"H2O AutoML R2 for {market} should be greater than 0.1"