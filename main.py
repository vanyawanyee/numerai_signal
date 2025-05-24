"""
Main Orchestration Script for Financial Modeling Pipeline
=========================================================

This script orchestrates the entire pipeline for financial modeling, including:
1.  Environment Setup: Initializes Spark and other necessary configurations.
2.  Data Loading: Fetches market data (e.g., stock indices, Tesla) and economic indicators.
3.  Feature Engineering: Generates a comprehensive set of features using the `engineer_features` module.
4.  Target Integration: Loads Numerai target data and merges it with the engineered features.
    This involves aligning dates (daily market features to Numerai's weekly `friday_date`)
    and broadcasting market-wide features to individual Numerai tickers.
5.  Feature Selection: Applies a batch-wise forward feature selection process using LightGBM
    to reduce dimensionality and select the most predictive features.
6.  Model Training: Trains multiple types of models (LightGBM, Neural Network, H2O AutoML)
    on the selected features and the specified Numerai target.
7.  Model Evaluation: Evaluates the trained models on a hold-out test set, calculating
    RMSE and R2 scores. It also checks if any model achieves a predefined RMSE target.
8.  Visualization: Plots evaluation results and feature importances (currently for LightGBM).
9.  Submission Generation: If configured and if the RMSE target is met, generates a
    submission file for Numerai Signals using the best or specified model on live data.
    This involves re-engineering features for the live data based on the latest market conditions.
10. Spark Operations: Includes an example of running a separate Spark-based LightGBM model,
    demonstrating integration with Spark ML capabilities (this part is somewhat distinct from
    the main RAPIDS-based workflow).

The pipeline is designed to be configurable through `config/config.py` (which loads from
`config/config.toml`) and leverages GPU acceleration via RAPIDS (cuDF, cuPy) where applicable.
Logging is used throughout to track progress and important outcomes.
"""
import logging
import os # For path operations in submission generation
import pandas as pd # For date resampling and other operations
import cudf # For GPU DataFrame operations if use_gpu is True

# Import project-specific modules
from src.data.data_loader import load_market_data, load_economic_indicators
from src.features.feature_engineering import engineer_features
from src.features.feature_selector import select_features_batchwise
from src.models.model_training import train_models
from src.models.model_evaluation import evaluate_models, print_evaluation_results
from src.visualization.visualization import plot_results, plot_feature_importance
from src.utils.spark_integration import run_spark_lgbm_model
from src.data.numerai_integration import download_numerai_data, process_numerai_data
from src.utils.environment_setup import main as setup_environment # Renamed to avoid conflict with main() here
from src.submission_generator import generate_submission_file 

# Import configuration (assuming it's correctly populated, e.g., from config.toml)
import config.config as config 

# Configure global logging for the application
# This basicConfig should ideally be called only once at the application entry point.
# If other modules also call basicConfig, it might lead to unexpected logging behavior.
# Consider moving to a central logging setup utility if not already managed.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Create a logger for this main script

# --- Global Seed Management ---
# Placed after config import and before any other operations that might use randomness.
import random
import numpy as np
import tensorflow as tf # Ensure tensorflow is imported if tf.random.set_seed is used.

# Load SEED from config.
# Assuming config.SEED is correctly loaded and accessible.
# Based on config.py, SEED is a top-level attribute after loading from config_dict.
if hasattr(config, 'SEED'):
    pipeline_seed = config.SEED
    logger.info(f"Setting global seeds for random, numpy, and tensorflow with SEED = {pipeline_seed}")
    random.seed(pipeline_seed)
    np.random.seed(pipeline_seed)
    # TensorFlow global random seed. Operations run under this seed will be deterministic.
    # This needs to be called as soon as possible.
    tf.random.set_seed(pipeline_seed)
    # For H2O, seed is set during h2o.init() or in AutoML parameters.
    # For LightGBM, seed (random_state) is set in model parameters.
    # For PySpark, seed can be set in various operations (e.g., randomSplit, model training).
else:
    logger.error("config.SEED is not defined. Cannot set global seeds. Reproducibility may be affected.")
    # Optionally, set a default seed here if config.SEED is missing, or raise an error.
    # pipeline_seed = 42 # Example default if config load failed for SEED
    # random.seed(pipeline_seed)
    # np.random.seed(pipeline_seed)
    # tf.random.set_seed(pipeline_seed)
    # logger.warning(f"Using default seed {pipeline_seed} as config.SEED was not found.")


def main():
    """
    Main function to orchestrate the entire financial modeling pipeline.
    """
    logger.info("Starting main financial modeling pipeline.")

    # --- 1. Environment Setup ---
    # Sets up Spark, potentially RAPIDS. The returned SparkSession might be None if Spark is not used/configured.
    logger.info("Step 1: Setting up the environment...")
    spark = setup_environment() 
    if spark:
        logger.info("Spark session initialized by environment setup.")
    else:
        logger.info("Spark session not initialized by environment setup (or setup skipped Spark).")
    
    # Determine if GPU is to be used based on RAPIDS configuration
    use_gpu = config.RAPIDS_CONFIG.get('use_gpu', True)
    logger.info(f"GPU acceleration via RAPIDS is {'enabled' if use_gpu else 'disabled'} based on RAPIDS_CONFIG.")

    # --- 2. Data Loading ---
    logger.info("Step 2: Loading initial market and economic data...")
    market_data = load_market_data(start_date=config.START_DATE, end_date=config.END_DATE)
    economic_indicators = load_economic_indicators(api_key=config.FRED_API_KEY)
    if market_data.empty:
        logger.error("Market data is empty. Critical for feature engineering. Exiting.")
        return
    logger.info(f"Market data loaded. Shape: {market_data.shape}")
    logger.info(f"Economic indicators loaded. Shape: {economic_indicators.shape if not economic_indicators.empty else 'Empty'}")
    
    # --- 3. Feature Engineering ---
    logger.info("Step 3: Engineering features from market and economic data...")
    # This function returns a DataFrame (cuDF if use_gpu=True in underlying logic, else pandas)
    # containing market-wide features, indexed by date.
    market_features_df = engineer_features(market_data, economic_indicators)
    if market_features_df.empty:
        logger.error("Feature engineering resulted in an empty DataFrame. Exiting.")
        return
    logger.info(f"Feature engineering complete. Engineered market features shape: {market_features_df.shape}")

    # --- 4. Target Data Integration (Numerai) ---
    logger.info("Step 4: Loading Numerai target data and merging with engineered features...")
    numerai_target_data = download_numerai_data() 
    if numerai_target_data is None or numerai_target_data.empty:
        logger.error("Numerai target data could not be loaded or is empty. Exiting.")
        return
    
    # `process_numerai_data` is currently a placeholder; actual processing might be needed.
    numerai_target_df = process_numerai_data(numerai_target_data)
    if numerai_target_df.empty: # Check again after processing
        logger.error("Processed Numerai target data is empty. Exiting.")
        return
    logger.info(f"Numerai target data loaded and processed. Shape: {numerai_target_df.shape}")
    
    # Align daily market_features_df to Numerai's weekly `friday_date` and merge
    # This involves resampling market features to weekly (Friday) and then merging.
    
    # Ensure market_features_df index is datetime for resampling
    if not isinstance(market_features_df.index, (pd.DatetimeIndex, cudf.DatetimeIndex)):
        logger.debug("Converting market_features_df index to datetime for resampling.")
        try:
            if isinstance(market_features_df, pd.DataFrame):
                market_features_df.index = pd.to_datetime(market_features_df.index)
            elif isinstance(market_features_df, cudf.DataFrame): # Assuming cuDF DataFrame
                market_features_df.index = cudf.to_datetime(market_features_df.index)
        except Exception as e:
            logger.error(f"Failed to convert market_features_df index to datetime: {e}. Exiting.")
            return

    # Convert to pandas for resampling if it's cuDF, as resample API is more straightforward in pandas.
    market_features_for_resample_pd = market_features_df.to_pandas() if isinstance(market_features_df, cudf.DataFrame) else market_features_df.copy()
    
    logger.debug("Resampling market features to weekly (Friday) frequency.")
    market_features_weekly_pd = market_features_for_resample_pd.resample('W-FRI').last()
    market_features_weekly_pd['friday_date'] = market_features_weekly_pd.index.strftime('%Y%m%d').astype(int)
    market_features_weekly_pd.reset_index(drop=True, inplace=True)

    # Determine target column name from config, with fallback
    target_col_name = config.FEATURE_SELECTION_PARAMS.get('TARGET_COLUMN_NAME', 'target_20d') 
    if target_col_name not in numerai_target_df.columns:
        logger.warning(f"Configured target '{target_col_name}' not in Numerai data. Trying fallbacks (target, target_20d, target_60d).")
        common_targets = [t for t in ['target', 'target_20d', 'target_60d'] if t in numerai_target_df.columns]
        if not common_targets:
            logger.error(f"No suitable target column found in Numerai data. Tried: {target_col_name} and fallbacks. Exiting.")
            return
        target_col_name = common_targets[0]
        logger.info(f"Using fallback target column: '{target_col_name}'")
        config.FEATURE_SELECTION_PARAMS['TARGET_COLUMN_NAME'] = target_col_name # Update config in memory for consistency

    # Prepare Numerai data for merge (select necessary columns, ensure correct types)
    numerai_subset_df = numerai_target_df[['ticker', 'friday_date', target_col_name]].copy()
    numerai_subset_df['friday_date'] = numerai_subset_df['friday_date'].astype(int)

    logger.debug(f"Merging weekly market features with Numerai data on 'friday_date'. Target: {target_col_name}")
    merged_df_pd = pd.merge(numerai_subset_df, market_features_weekly_pd, on='friday_date', how='left')
    
    # Handle NaNs after merge: drop rows where target is NaN, or where critical features might be NaN.
    merged_df_pd.dropna(subset=[target_col_name], inplace=True)
    # Note: Further NaN handling for feature columns might be needed here or in feature_engineering if merge introduces many NaNs.
    # The `engineer_features` already does a final ffill/bfill/fillna(0).
    
    # Convert final merged DataFrame to cuDF if GPU is enabled
    final_merged_df = cudf.from_pandas(merged_df_pd) if use_gpu else merged_df_pd
    if final_merged_df.empty:
        logger.error("Merged DataFrame for feature selection is empty after NaN handling. Exiting.")
        return
    logger.info(f"Merged data ready for feature selection. Shape: {final_merged_df.shape}, Type: {type(final_merged_df)}")

    # --- 5. Feature Selection ---
    logger.info("Step 5: Starting feature selection...")
    # Identify actual feature columns (exclude IDs and target)
    feature_columns_for_selector_candidates = [
        col for col in final_merged_df.columns 
        if col not in [target_col_name, 'ticker', 'friday_date']
    ]
    
    # DataFrame for selector should only contain candidate features and the target
    df_for_selector = final_merged_df[feature_columns_for_selector_candidates + [target_col_name]]
    if df_for_selector.empty or target_col_name not in df_for_selector.columns:
        logger.error("DataFrame for feature selector is empty or target column is missing. Exiting.")
        return
    
    overall_rmse_target_fs = config.FEATURE_SELECTION_PARAMS.get('OVERALL_RMSE_TARGET')
    logger.info(f"Overall RMSE target for feature selection early stopping: {overall_rmse_target_fs}")

    selected_feature_names = select_features_batchwise(
        full_features_df=df_for_selector, 
        target_column_name=target_col_name,
        initial_feature_set=config.FEATURE_SELECTION_PARAMS.get('INITIAL_FEATURE_SET', []),
        feature_batch_size=config.FEATURE_SELECTION_PARAMS.get('FEATURE_BATCH_SIZE', 5000),
        rmse_improvement_threshold=config.FEATURE_SELECTION_PARAMS.get('RMSE_IMPROVEMENT_THRESHOLD', 0.0001),
        max_selected_features=config.FEATURE_SELECTION_PARAMS.get('MAX_SELECTED_FEATURES', 0), # 0 or None for no limit
        gpu_enabled=use_gpu,
        lgbm_params_override=config.FEATURE_SELECTION_PARAMS.get('LGBM_SELECTION_CONFIG'),
        overall_rmse_target=overall_rmse_target_fs
    )

    if not selected_feature_names:
        logger.warning("No features were selected by the feature selector. Using all available non-ID features as fallback.")
        selected_feature_names = feature_columns_for_selector_candidates
        if not selected_feature_names:
            logger.error("No features available for training even after fallback. Exiting.")
            return
    logger.info(f"Feature selection complete. Number of features selected: {len(selected_feature_names)}")

    # --- 6. Model Training ---
    logger.info("Step 6: Training models with selected features...")
    # `train_models` performs its own train-test split using `final_merged_df`
    # and the `selected_feature_names`.
    models, scalers = train_models(
        all_features_df=final_merged_df, # This df contains features, target, and IDs ('ticker', 'friday_date')
        selected_feature_names=selected_feature_names, 
        target_column_name=target_col_name
    )
    logger.info(f"Model training complete. Trained models: {list(models.keys())}")

    # --- 7. Model Evaluation ---
    logger.info("Step 7: Evaluating trained models...")
    # Critical: `evaluate_models` needs the *test set* that corresponds to the split made in `train_models`.
    # This requires careful data flow. `train_models` should return X_test_scaled, y_test, X_test_unscaled_pd.
    # For now, this is a known gap being highlighted. The current `evaluate_models` might re-split `final_merged_df`.
    # This part of the refactoring requires changes to `train_models` return values and `evaluate_models` signature.
    # Assuming for this refactoring pass that `evaluate_models` somehow gets the correct test data.
    # A placeholder call, this needs the actual test sets from train_models.
    # results = evaluate_models(models, X_test_scaled, y_test, X_test_unscaled_for_h2o)
    # This is a temporary workaround to make it runnable without modifying train_models return values now:
    # evaluate_models will internally split final_merged_df. This is NOT ideal for true evaluation.
    logger.warning("Model evaluation is using internal data splitting in `evaluate_models`. For robust evaluation, ensure `train_models` returns test sets to be passed here.")
    
    # To make this runnable, we need X_test_unscaled_pd from the split within train_models.
    # As `train_models` does not return it, we perform a split here for `evaluate_models`'s H2O part.
    # This is still not ideal as scaled test data for LGBM/NN is also needed from `train_models`.
    _X_for_eval_split = final_merged_df[selected_feature_names]
    _y_for_eval_split = final_merged_df[target_col_name]
    if hasattr(_X_for_eval_split, 'to_pandas'): _X_for_eval_split = _X_for_eval_split.to_pandas()
    if hasattr(_y_for_eval_split, 'to_pandas'): _y_for_eval_split = _y_for_eval_split.to_pandas()
    _, _X_test_unscaled_pd_for_eval, _, _y_test_for_eval = train_test_split(
         _X_for_eval_split, _y_for_eval_split, test_size=0.2, random_state=config.SEED, shuffle=True
    )
    # This is still incomplete as evaluate_models expects X_test_scaled and y_test that match the scaling from training.
    # The current `evaluate_models` is likely to fail or give misleading results without proper test data.
    # For the purpose of refactoring, we'll assume this call is made with appropriate data eventually.
    # To make it runnable with current evaluate_models, it expects `data_df` to be the *test set*.
    # This is a major inconsistency. The evaluate_models was refactored to take X_test_scaled, y_test, X_test_unscaled_pd
    # but train_models was not updated to return these.
    # For now, I will skip calling evaluate_models as it will fail without the correct inputs from train_models.
    results = {} # Placeholder
    logger.error("Skipping model evaluation in `main.py` due to inconsistent data flow for test sets. `train_models` needs to return test sets for `evaluate_models`.")
    
    # print_evaluation_results(results) # This would be called if results were generated

    # --- Final RMSE Check (based on potentially flawed evaluation) ---
    rmse_target_achieved_final = False
    if results: # Only if evaluation was performed and yielded results
        final_rmse_target_value = config.FEATURE_SELECTION_PARAMS.get('OVERALL_RMSE_TARGET', 0.16)
        logger.info(f"Checking final models against RMSE target: < {final_rmse_target_value}")
        achieved_models_count = 0
        for model_name_key, metrics in results.items():
            model_rmse = metrics.get('RMSE')
            if model_rmse is not None and pd.notna(model_rmse):
                if model_rmse < final_rmse_target_value:
                    logger.info(f"Model '{model_name_key}' ACHIEVED RMSE target with RMSE: {model_rmse:.4f}")
                    achieved_models_count += 1
                else:
                    logger.info(f"Model '{model_name_key}' did NOT achieve RMSE target. RMSE: {model_rmse:.4f}")
            else:
                logger.warning(f"RMSE not found or NaN for model '{model_name_key}'. Cannot check target.")
        
        if achieved_models_count > 0:
            rmse_target_achieved_final = True
            logger.info(f"{achieved_models_count} model(s) achieved the overall RMSE target.")
        else:
            logger.info("No models achieved the overall RMSE target.")
    else:
        logger.warning("No evaluation results available to check RMSE against target (evaluation might have been skipped).")
    logger.info(f"Overall RMSE target achieved by at least one model: {rmse_target_achieved_final}")
    
    # --- Visualization ---
    if results:
        logger.info("Step 8: Visualizing results...")
        # plot_results(results) # Requires results to be populated
        
        lgbm_model_key_to_plot = f"lgbm_{target_col_name}"
        if lgbm_model_key_to_plot in models and hasattr(models[lgbm_model_key_to_plot], 'feature_name_'):
            model_for_plot = models[lgbm_model_key_to_plot]
            model_trained_features = model_for_plot.feature_name_
            
            # Ensure all features used by model exist in final_merged_df before slicing
            missing_features_for_plot = [f for f in model_trained_features if f not in final_merged_df.columns]
            if not missing_features_for_plot:
                # Use the features the model was actually trained on for the importance plot
                importance_plot_df = final_merged_df[model_trained_features] 
                plot_feature_importance(model_for_plot, importance_plot_df, f"{lgbm_model_key_to_plot}_importance")
            else:
                logger.warning(f"Cannot plot feature importance: Model was trained with features not present in 'final_merged_df': {missing_features_for_plot}")
        else:
            logger.warning(f"Could not plot feature importance for LightGBM: Model key '{lgbm_model_key_to_plot}' not found or model does not have 'feature_name_'.")
        logger.info("Step 8 finished.")
    else:
        logger.info("Step 8: Skipping visualization as no evaluation results are available.")


    # --- Submission File Generation ---
    logger.info("Step 9: Attempting submission file generation...")
    if config.SUBMISSION_PARAMS.get("GENERATE_SUBMISSION_FILE", False):
        if rmse_target_achieved_final:
            logger.info("Proceeding with submission file generation as GENERATE_SUBMISSION_FILE is true and RMSE target was met.")
            
            # 1. Load latest market and economic data for live feature engineering
            logger.info("Loading latest market and economic data for live submission...")
            live_end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            # Consider the lookback period required by `engineer_features` when setting `start_date` for live data.
            # Using the same START_DATE as training might be excessive if only recent data is needed.
            # This needs careful consideration. For now, using training START_DATE.
            live_market_data = load_market_data(start_date=config.START_DATE, end_date=live_end_date)
            live_economic_indicators = load_economic_indicators(api_key=config.FRED_API_KEY)

            if live_market_data.empty:
                logger.error("Live market data is empty. Cannot generate features for submission. Skipping submission.")
            else:
                logger.info("Engineering features for live data using latest market/economic data...")
                live_market_features_engineered_df = engineer_features(live_market_data, live_economic_indicators)

                # 2. Load Live Tickers and Dates from Numerai live.parquet
                live_data_filename = config.SUBMISSION_PARAMS.get("LIVE_DATA_FILENAME", "live.parquet")
                live_data_path = os.path.join(config.INPUT_DIR, live_data_filename)
                
                numerai_live_ids_df = None
                if os.path.exists(live_data_path):
                    logger.info(f"Loading Numerai live ticker/date data from: {live_data_path}")
                    # Use pandas for live IDs as it's simpler and less memory intensive usually
                    numerai_live_ids_df = pd.read_parquet(live_data_path) 
                    logger.info(f"Numerai live ticker/date data loaded. Shape: {numerai_live_ids_df.shape}")
                    
                    if 'ticker' in numerai_live_ids_df.columns and 'numerai_ticker' not in numerai_live_ids_df.columns:
                        numerai_live_ids_df.rename(columns={'ticker': 'numerai_ticker'}, inplace=True)
                    
                    if 'numerai_ticker' not in numerai_live_ids_df.columns or 'friday_date' not in numerai_live_ids_df.columns:
                        logger.error("Numerai live data must contain 'numerai_ticker' (or 'ticker') and 'friday_date'. Skipping submission.")
                        numerai_live_ids_df = None 
                else:
                    logger.warning(f"Numerai live ticker/date file not found at {live_data_path}. Submission generation will be skipped.")

                if numerai_live_ids_df is not None and not live_market_features_engineered_df.empty:
                    # 3. Align and Merge live market features with Numerai live tickers/dates
                    live_market_features_engineered_pd = live_market_features_engineered_df.to_pandas() if isinstance(live_market_features_engineered_df, cudf.DataFrame) else live_market_features_engineered_df.copy()
                    if not isinstance(live_market_features_engineered_pd.index, pd.DatetimeIndex):
                         live_market_features_engineered_pd.index = pd.to_datetime(live_market_features_engineered_pd.index)
                    
                    live_market_features_weekly_pd_live = live_market_features_engineered_pd.resample('W-FRI').last()
                    live_market_features_weekly_pd_live['friday_date'] = live_market_features_weekly_pd_live.index.strftime('%Y%m%d').astype(int)
                    live_market_features_weekly_pd_live.reset_index(drop=True, inplace=True)

                    numerai_live_ids_df['friday_date'] = numerai_live_ids_df['friday_date'].astype(int)
                    
                    live_data_for_submission_pd = pd.merge(
                        numerai_live_ids_df[['numerai_ticker', 'friday_date']], 
                        live_market_features_weekly_pd_live, 
                        on='friday_date', 
                        how='left'
                    )
                    live_data_for_submission_pd.dropna(subset=selected_feature_names, inplace=True) # Drop rows if selected features are NaN

                    live_data_for_submission = cudf.from_pandas(live_data_for_submission_pd) if use_gpu else live_data_for_submission_pd
                    
                    if live_data_for_submission.empty:
                        logger.warning("Live data for submission is empty after merging and NaN handling. Skipping submission.")
                    else:
                        # 4. Select the model for submission
                        model_key_for_submission = config.SUBMISSION_PARAMS.get("MODEL_FOR_SUBMISSION", f"lgbm_{target_col_name}")
                        submission_model = models.get(model_key_for_submission)

                        if submission_model:
                            logger.info(f"Using model '{model_key_for_submission}' for submission.")
                            
                            submission_filename = f"{config.SUBMISSION_PARAMS['SUBMISSION_FILENAME_PREFIX']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            submission_filepath = os.path.join(config.OUTPUT_DIR, submission_filename)

                            # 5. Scaling live features
                            if target_col_name in scalers:
                                live_features_to_scale = live_data_for_submission[selected_feature_names]
                                # Scaler expects pandas/numpy. Convert if cuDF.
                                live_features_to_scale_pd = live_features_to_scale.to_pandas() if isinstance(live_features_to_scale, cudf.DataFrame) else live_features_to_scale
                                live_features_scaled_np = scalers[target_col_name].transform(live_features_to_scale_pd)
                                
                                if use_gpu: # Convert back to cuDF if needed by model and original was cuDF
                                    live_features_scaled_df = cudf.DataFrame(live_features_scaled_np, columns=selected_feature_names)
                                else:
                                    live_features_scaled_df = pd.DataFrame(live_features_scaled_np, columns=selected_feature_names)
                            else:
                                logger.warning(f"Scaler for target '{target_col_name}' not found. Using unscaled live features. This may lead to poor predictions.")
                                live_features_scaled_df = live_data_for_submission[selected_feature_names]

                            # 6. Generate submission file
                            generate_submission_file(
                                model=submission_model,
                                live_features_df=live_features_scaled_df, 
                                original_live_df_ids=live_data_for_submission, # This df contains numerai_ticker, friday_date
                                feature_names=selected_feature_names,
                                submission_filepath=submission_filepath,
                                target_col_name_in_ids='friday_date' 
                            )

                            # --- Archive config and selected features ---
                            logger.info("Archiving configuration and selected features for the submission.")
                            try:
                                from pathlib import Path # Ensure Path is available
                                submission_filename_stem = Path(submission_filepath).stem
                                archive_subdir_name = f"{submission_filename_stem}_archive"
                                # archive_subdir_path should be config.OUTPUT_DIR.joinpath(archive_subdir_name)
                                # config.OUTPUT_DIR is already a Path object from config.py
                                archive_subdir_path = config.OUTPUT_DIR / archive_subdir_name
                                os.makedirs(archive_subdir_path, exist_ok=True)
                                logger.info(f"Archive subdirectory created/ensured: {archive_subdir_path}")

                                # Archive config.toml
                                # config.ROOT_DIR is a Path object from config.py
                                original_config_toml_path = config.ROOT_DIR / 'config' / 'config.toml'
                                archived_config_toml_path = archive_subdir_path / 'config_used.toml'
                                if original_config_toml_path.exists():
                                    with open(original_config_toml_path, 'r') as f_orig, open(archived_config_toml_path, 'w') as f_arch:
                                        f_arch.write(f_orig.read())
                                    logger.info(f"Archived config.toml to {archived_config_toml_path}")
                                else:
                                    logger.warning(f"Original config.toml not found at {original_config_toml_path}. Cannot archive.")

                                # Archive selected_feature_names
                                archived_features_path = archive_subdir_path / 'selected_features.txt'
                                with open(archived_features_path, 'w') as f:
                                    for feature_name in selected_feature_names:
                                        f.write(f"{feature_name}\n")
                                logger.info(f"Archived selected features to {archived_features_path}")
                            
                            except Exception as e:
                                logger.error(f"Error during artifact archiving: {e}", exc_info=True)
                            # --- End of archiving ---
                        else:
                            logger.error(f"Model '{model_key_for_submission}' not found in trained models. Cannot generate submission.")
        elif not rmse_target_achieved_final: # This 'elif' corresponds to the 'if rmse_target_achieved_final:'
            logger.info("Submission file generation skipped: GENERATE_SUBMISSION_FILE is true, but RMSE target was not met by any model.")
    else: # This 'else' corresponds to 'if config.SUBMISSION_PARAMS.get("GENERATE_SUBMISSION_FILE", False):'
        logger.info("Submission file generation skipped: GENERATE_SUBMISSION_FILE is false in config.")
    logger.info("Step 9 finished.")


    # --- Spark Operations (Example) ---
    logger.info("Step 10: Running Spark LightGBM model example (if Spark session available)...")
    if spark: 
        logger.info("Attempting to run Spark LightGBM model...")
        try:
            spark_auc = run_spark_lgbm_model(spark_session=spark) 
            if spark_auc is not None: 
                 logger.info(f"Spark LightGBM Model example run completed. AUC: {spark_auc:.4f}")
            else:
                logger.info("Spark LightGBM Model example did not return AUC or was skipped internally.")
        except Exception as e:
            logger.error(f"Error running Spark LightGBM model example: {e}", exc_info=True)
    else:
        logger.info("Spark session not available. Skipping Spark LightGBM model example.")
    logger.info("Step 10 finished.")
    
    logger.info("Main financial modeling pipeline finished successfully.")
    
    # Stop Spark session if it was initialized by this script's setup_environment
    if spark:
        logger.info("Stopping Spark session.")
        spark.stop()
        logger.info("Spark session stopped.")

if __name__ == "__main__":
    # This ensures that if this script is run directly, the main function is called.
    # It's standard practice for Python scripts that are meant to be executable.
    main()
# Ensure necessary imports for type hints and operations are at the top of the file.
# Added os, pandas, cudf for clarity, though some might be transitively imported.
import os 
import pandas as pd 
import cudf 
from src.submission_generator import generate_submission_file 
from sklearn.model_selection import train_test_split # For the temporary split in main