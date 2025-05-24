"""
Feature Selection Module
========================

This module provides functionality for selecting the most relevant features from a 
large feature set. The primary method implemented is batch-wise forward selection 
using LightGBM as the evaluation model. This approach is designed to be scalable 
for high-dimensional data and can leverage GPU acceleration if available.

Key Components:
- `train_lgbm_and_get_rmse`: A helper function to train a LightGBM model with a
  subset of features and evaluate its performance (RMSE) on a validation set.
  It handles data type conversions and GPU configuration for LightGBM.
- `select_features_batchwise`: The main feature selection function. It iteratively
  adds batches of candidate features to an existing set of selected features,
  evaluates them, and keeps a batch if it improves the model's RMSE beyond a
  specified threshold. It also supports early stopping if an overall RMSE target
  is met or a maximum number of features is selected.

The selection process is crucial for reducing model complexity, improving training
speed, and potentially enhancing model generalization by removing noisy or
redundant features.
"""
import cudf
import pandas as pd
import lightgbm as lgb
import numpy as np
import time
import logging # For logging information and warnings
from sklearn.model_selection import train_test_split as sk_train_test_split # For splitting if not already done
# Import cupy if it's used for RMSE calculation, ensure it's handled if not available
try:
    import cupy as cp
except ImportError:
    cp = None 
    # logger.info("CuPy not found. RMSE calculation will use NumPy.") # Logger not defined yet at module level

# Configure logger for this module
logger = logging.getLogger(__name__)

# Import global SEED from config
from config.config import SEED as global_pipeline_seed

def train_lgbm_and_get_rmse(X_train, y_train, X_val, y_val, features_to_use, gpu_enabled=True, lgbm_params=None):
    """
    Trains a LightGBM model with a given set of features and returns the validation RMSE.

    Args:
        X_train: Training feature DataFrame (pandas or cuDF).
        y_train: Training target Series (pandas or cuDF).
        X_val: Validation feature DataFrame (pandas or cuDF).
        y_val: Validation target Series (pandas or cuDF).
        features_to_use (list): List of column names to use as features.
        gpu_enabled (bool): Whether to enable GPU for LightGBM.
        lgbm_params (dict, optional): Custom LightGBM parameters to override defaults.

    Returns:
        float: The calculated RMSE on the validation set.
    """
    logger.debug(f"Training LGBM with {len(features_to_use)} features. GPU enabled: {gpu_enabled}")

    # Ensure data is in float32 for LightGBM, especially if using cuDF for GPU training
    # This also handles the selection of `features_to_use`
    if isinstance(X_train, cudf.DataFrame):
        X_train_lgb = X_train[features_to_use].astype(np.float32)
        y_train_lgb = y_train.astype(np.float32)
        X_val_lgb = X_val[features_to_use].astype(np.float32)
        y_val_lgb = y_val.astype(np.float32)
    elif isinstance(X_train, pd.DataFrame):
        X_train_lgb = X_train[features_to_use].astype(np.float32)
        y_train_lgb = y_train.astype(np.float32)
        X_val_lgb = X_val[features_to_use].astype(np.float32)
        y_val_lgb = y_val.astype(np.float32)
    else:
        # This case might occur if data is already a NumPy/CuPy array; ensure selection is done.
        # However, the main function `select_features_batchwise` prepares pandas/cuDF DataFrames.
        logger.warning("Input data to train_lgbm_and_get_rmse is not pandas/cuDF. Direct usage of features_to_use assumed.")
        X_train_lgb = X_train[features_to_use] if isinstance(X_train, (np.ndarray, cp.ndarray if cp else np.ndarray)) else X_train # Basic selection for arrays
        y_train_lgb = y_train
        X_val_lgb = X_val[features_to_use] if isinstance(X_val, (np.ndarray, cp.ndarray if cp else np.ndarray)) else X_val
        y_val_lgb = y_val
        # Note: Further type casting might be needed here if array inputs are not float32.

    # Default LightGBM parameters for feature selection - focused on speed and generalization
    default_params = {
        'objective': 'regression_l1', # MAE objective, often more robust to outliers than L2 (MSE)
        'metric': 'rmse',             # Evaluation metric
        'n_estimators': 200,          # Number of boosting rounds (relatively low for speed in selection)
        'learning_rate': 0.05,
        'feature_fraction': 0.8,      # Equivalent to colsample_bytree
        'bagging_fraction': 0.8,      # Equivalent to subsample
        'bagging_freq': 1,            # Perform bagging at every 1 iteration
        'verbose': -1,                # Suppress LightGBM's own console output
        'n_jobs': -1,                 # Use all available cores for CPU training
        'seed': global_pipeline_seed, # Use global SEED for reproducibility
    }
    if gpu_enabled:
        default_params['device'] = 'gpu'
        # default_params['gpu_platform_id'] = 0 # Typically auto-detected
        # default_params['gpu_device_id'] = 0   # Typically auto-detected, or uses all available with data_parallel
        # LightGBM's data_parallel tree learner can use multiple GPUs if available.
        default_params['tree_learner'] = 'data_parallel' 
        logger.debug("LGBM GPU parameters enabled.")


    if lgbm_params: # Override defaults with any custom parameters provided
        default_params.update(lgbm_params)
    logger.debug(f"LGBM effective parameters: {default_params}")

    model = lgb.LGBMRegressor(**default_params)
    
    try:
        model.fit(X_train_lgb, y_train_lgb,
                  eval_set=[(X_val_lgb, y_val_lgb)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(50, verbose=False, min_delta=0.00001)]) # Early stopping if no improvement
    except Exception as e:
        logger.error(f"Error during LightGBM model fitting: {e}")
        logger.error(f"X_train shape: {X_train_lgb.shape}, y_train shape: {y_train_lgb.shape}")
        logger.error(f"X_val shape: {X_val_lgb.shape}, y_val shape: {y_val_lgb.shape}")
        # logger.error(f"X_train dtypes: {X_train_lgb.dtypes if hasattr(X_train_lgb, 'dtypes') else X_train_lgb.dtype}")
        # logger.error(f"y_train dtype: {y_train_lgb.dtype}")
        raise # Re-raise the exception after logging

    preds = model.predict(X_val_lgb)
    
    # Calculate RMSE, handling different array types (pandas, cuDF, numpy, cupy)
    y_val_values = y_val_lgb
    if hasattr(y_val_lgb, 'values'): # Handles pandas/cuDF Series by getting underlying array
        y_val_values = y_val_lgb.values

    rmse_val = 0.0
    if cp and 'cupy' in str(type(y_val_values)): # If y_val_values is a CuPy array
        preds_cp = cp.asarray(preds) # Ensure predictions are also CuPy array
        rmse_val = float(cp.sqrt(cp.mean((y_val_values - preds_cp)**2)))
    elif isinstance(y_val_values, np.ndarray) or isinstance(y_val_lgb, pd.Series): 
        # If y_val_values is NumPy array, or y_val_lgb was a pandas Series (its .values is NumPy)
        rmse_val = np.sqrt(np.mean((np.asarray(y_val_values) - np.asarray(preds))**2))
    else: 
        logger.warning(f"y_val_lgb type {type(y_val_lgb)} not explicitly handled for RMSE, attempting generic numpy conversion.")
        try:
            rmse_val = np.sqrt(np.mean((np.array(y_val_lgb) - np.asarray(preds))**2))
        except Exception as e:
            logger.error(f"Could not calculate RMSE due to type issues: {e}")
            return float('inf') # Return a high RMSE if calculation fails
            
    logger.debug(f"LGBM training complete. Validation RMSE: {rmse_val:.5f}")
    return rmse_val

def select_features_batchwise(
    full_features_df, 
    target_column_name, 
    initial_feature_set=None, 
    feature_batch_size=500, 
    rmse_improvement_threshold=0.001, 
    max_selected_features=None, 
    gpu_enabled=True,
    val_size=0.2, 
    random_state_split=global_pipeline_seed, # Use global SEED here too for the split if shuffle=True
    lgbm_params_override=None, 
    overall_rmse_target=None 
    ):
    """
    Selects features using a batch-wise forward selection approach with LightGBM.

    This method iteratively adds batches of candidate features to a core set of
    selected features. After adding each batch, a LightGBM model is trained, and
    its RMSE on a validation set is evaluated. The batch is permanently added to
    the selected set only if it improves the RMSE by at least `rmse_improvement_threshold`.

    The process can stop early under three conditions:
    1. No more candidate features are left.
    2. The `max_selected_features` limit is reached.
    3. The `overall_rmse_target` is achieved by the model.

    Args:
        full_features_df (pd.DataFrame or cudf.DataFrame): DataFrame containing all features and the target column.
        target_column_name (str): Name of the target variable column.
        initial_feature_set (list, optional): A list of feature names to start with. Defaults to empty.
        feature_batch_size (int): Number of new features to evaluate in each batch.
        rmse_improvement_threshold (float): Minimum RMSE improvement required to keep a new batch of features.
        max_selected_features (int, optional): Maximum number of features to select. If None, no limit.
        gpu_enabled (bool): Whether to use GPU for LightGBM training during selection.
        val_size (float): Proportion of the data to use for the validation set.
        random_state_split (int): Random seed for the train-validation split (used if shuffle=True).
        lgbm_params_override (dict, optional): Custom LightGBM parameters for the selection models.
                                               Should also include 'seed' or 'random_state' if overriding.
        overall_rmse_target (float, optional): If the model's RMSE drops below this value, selection stops early.

    Returns:
        list: A list of selected feature names.
    """
    start_time = time.time()
    logger.info("Starting batch-wise forward feature selection process...")

    if initial_feature_set is None:
        initial_feature_set = []

    if not isinstance(full_features_df, (pd.DataFrame, cudf.DataFrame)):
        logger.error("full_features_df must be a pandas or cuDF DataFrame.")
        raise ValueError("full_features_df must be a pandas or cuDF DataFrame.")

    # Handle data type conversion based on gpu_enabled flag
    if gpu_enabled and isinstance(full_features_df, pd.DataFrame):
        logger.info("GPU enabled: Converting pandas DataFrame to cuDF for feature selection.")
        full_features_df = cudf.from_pandas(full_features_df)
    elif not gpu_enabled and isinstance(full_features_df, cudf.DataFrame):
        logger.info("GPU disabled: Converting cuDF DataFrame to pandas for feature selection.")
        full_features_df = full_features_df.to_pandas()
    
    if target_column_name not in full_features_df.columns:
        logger.error(f"Target column '{target_column_name}' not found in DataFrame.")
        raise ValueError(f"Target column '{target_column_name}' not found in DataFrame.")

    X = full_features_df.drop(columns=[target_column_name])
    y = full_features_df[target_column_name]
    logger.info(f"Feature selection: X shape: {X.shape}, y shape: {y.shape}")

    # Data Splitting:
    # For time-series data, a chronological split (e.g., older data for train, newer for val) is crucial.
    # The current implementation uses sklearn's train_test_split with shuffle=False,
    # which is suitable if the DataFrame is already sorted by time.
    # If cuDF is used and cuML is available, cuML's train_test_split could be an alternative.
    logger.debug(f"Splitting data for validation. Validation size: {val_size}, Shuffle: False (important for time-series).")
    
    # Convert to pandas for splitting if cuDF, to ensure consistent splitting logic or if cuML is not used.
    # This conversion can be a performance consideration for very large cuDF DataFrames.
    X_for_split = X.to_pandas() if isinstance(X, cudf.DataFrame) else X
    y_for_split = y.to_pandas() if isinstance(y, cudf.DataFrame) else y
    
    # If shuffle were True, random_state_split (now global_pipeline_seed) would be used.
    # Since shuffle=False for time-series, random_state here doesn't affect row selection but is good practice to set.
    X_train_pd, X_val_pd, y_train_pd, y_val_pd = sk_train_test_split(
        X_for_split, y_for_split, 
        test_size=val_size, 
        shuffle=False, 
        random_state=random_state_split 
    )

    # Convert split data back to cuDF if GPU is enabled
    if gpu_enabled:
        X_train = cudf.from_pandas(X_train_pd)
        X_val = cudf.from_pandas(X_val_pd)
        y_train = cudf.from_pandas(y_train_pd)
        y_val = cudf.from_pandas(y_val_pd)
        # Ensure target Series have names, useful for some operations or debugging
        y_train.name = target_column_name 
        y_val.name = target_column_name
    else:
        X_train, X_val, y_train, y_val = X_train_pd, X_val_pd, y_train_pd, y_val_pd

    logger.info(f"Data split complete. X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # Initialize selected features and baseline RMSE
    selected_features = list(set(initial_feature_set).intersection(set(X.columns))) # Ensure initial features are valid
    
    if not selected_features: 
        logger.info("Initial feature set is empty or contains no valid features. Baseline RMSE will be infinity.")
        best_rmse = float('inf') # No baseline model if no initial features
    else:
        logger.info(f"Training baseline model with {len(selected_features)} initial features: {selected_features}")
        best_rmse = train_lgbm_and_get_rmse(X_train, y_train, X_val, y_val, selected_features, gpu_enabled, lgbm_params_override)
        logger.info(f"Baseline RMSE with initial features: {best_rmse:.5f}")

    # Prepare candidate features (all features not in the initial selected set)
    candidate_features = [f for f in X.columns if f not in selected_features]
    
    # Optional: Shuffle candidate features to mitigate selection bias from column ordering.
    # np.random.seed(random_state_split) # Ensure reproducibility if shuffling
    # np.random.shuffle(candidate_features) 
    # logger.debug("Candidate features shuffled (if implemented).")

    # Create batches of candidate features
    feature_batches = [candidate_features[i:i + feature_batch_size] 
                       for i in range(0, len(candidate_features), feature_batch_size)]
    logger.info(f"Starting batch feature selection with {len(feature_batches)} batches of size up to {feature_batch_size}.")

    # Iterate through batches, adding them to the selected set if RMSE improves
    for i, batch in enumerate(feature_batches):
        if not batch: # Skip empty batches (can happen if candidate_features count is not a multiple of batch_size)
            logger.debug(f"Batch {i+1}/{len(feature_batches)} is empty. Skipping.")
            continue
        
        current_trial_features = selected_features + batch # Features to evaluate in this iteration
        
        logger.info(f"\nEvaluating Batch {i+1}/{len(feature_batches)}: Adding {len(batch)} new features. Total trial features: {len(current_trial_features)}")
        
        current_rmse = train_lgbm_and_get_rmse(X_train, y_train, X_val, y_val, current_trial_features, gpu_enabled, lgbm_params_override)
        
        logger.info(f"Batch {i+1} evaluated. RMSE: {current_rmse:.5f}. Previous best RMSE: {best_rmse:.5f}")

        improvement = best_rmse - current_rmse # Positive value indicates improvement

        if improvement > rmse_improvement_threshold:
            selected_features.extend(batch) # Permanently add the batch of features
            best_rmse = current_rmse
            logger.info(f"Batch {i+1} KEPT. Improvement: {improvement:.5f}. New best RMSE: {best_rmse:.5f}. Total selected features: {len(selected_features)}")
            
            # Check for overall RMSE target achievement after improvement
            if overall_rmse_target is not None and best_rmse < overall_rmse_target:
                logger.info(f"Overall RMSE target ({overall_rmse_target}) achieved with current RMSE: {best_rmse:.5f}. Stopping feature selection early.")
                break # Exit the loop over batches
        else:
            logger.info(f"Batch {i+1} DISCARDED. RMSE did not improve sufficiently (Improvement: {improvement:.5f} <= Threshold: {rmse_improvement_threshold}).")
            # Special handling for the very first batch if initial_feature_set was empty:
            if i == 0 and not initial_feature_set and not selected_features: 
                # If no initial features, the first batch's RMSE becomes the baseline.
                # This batch is accepted to start the selection process.
                selected_features.extend(batch)
                best_rmse = current_rmse
                logger.info(f"First batch accepted to initialize baseline (as initial set was empty). RMSE: {best_rmse:.5f}. Total selected: {len(selected_features)}")
                # Check if this first batch already met the overall RMSE target
                if overall_rmse_target is not None and best_rmse < overall_rmse_target:
                    logger.info(f"Overall RMSE target ({overall_rmse_target}) achieved with first batch. Stopping feature selection.")
                    break # Exit the loop over batches

        # Check for max_selected_features limit
        if max_selected_features and len(selected_features) >= max_selected_features:
            logger.info(f"Reached maximum selected features limit ({max_selected_features}). Stopping feature selection.")
            break
            
    end_time = time.time()
    logger.info(f"\nFeature selection completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Total features selected: {len(selected_features)}. Final RMSE with selected features: {best_rmse:.5f}")
    logger.debug(f"Final selected features list: {selected_features}")
    
    return selected_features

if __name__ == '__main__':
    # Example Usage (requires dummy data generation)
    
    # Generate dummy data (pandas or cuDF)
    N_ROWS = 2000
    N_FEATURES = 10000 # Reduced for quick local test, original is ~200k
    N_INITIAL_FEATURES = 5
    TARGET_NAME = 'target'

    # Use pandas for simplicity in this example script
    # In the main pipeline, this would come from feature_engineering.py
    use_cudf_for_dummy = False # Set to True to test with cuDF dummy data

    if use_cudf_for_dummy:
        try:
            import cupy as cp
            cp.random.seed(0)
            dummy_data = cudf.DataFrame({f'feature_{i}': cp.random.rand(N_ROWS) for i in range(N_FEATURES)})
            dummy_data[TARGET_NAME] = cp.random.rand(N_ROWS)
            # Ensure some initial features have predictive power for a more meaningful test
            for i in range(N_INITIAL_FEATURES):
                 dummy_data[TARGET_NAME] += 2 * dummy_data[f'feature_{i}']
            print("Using cuDF for dummy data.")
        except ImportError:
            print("cuDF/cuPy not available, using pandas for dummy data.")
            use_cudf_for_dummy = False # Fallback to pandas

    if not use_cudf_for_dummy:
        np.random.seed(0)
        dummy_data = pd.DataFrame({f'feature_{i}': np.random.rand(N_ROWS) for i in range(N_FEATURES)})
        dummy_data[TARGET_NAME] = np.random.rand(N_ROWS)
        for i in range(N_INITIAL_FEATURES):
             dummy_data[TARGET_NAME] += 2 * dummy_data[f'feature_{i}']
   
    initial_feats = [f'feature_{i}' for i in range(N_INITIAL_FEATURES)]
    
    # Example LightGBM parameters for selection phase (can be tuned)
    lgbm_selection_params = {
        'n_estimators': 100, 
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'verbose': -1,
    }

    selected = select_features_batchwise(
        full_features_df=dummy_data,
        target_column_name=TARGET_NAME,
        initial_feature_set=initial_feats,
        feature_batch_size=1000, # How many new features to test at a time
        rmse_improvement_threshold=-0.0001, # Allow slight worsening to explore, or set to 0 or positive
        max_selected_features=2000, # Max number of features to end up with
        gpu_enabled=False, # Set to True if LightGBM GPU is available and desired
        lgbm_params_override=lgbm_selection_params
    )
    print("\n--- Example Run Finished ---")
    # To run this example: python src/features/feature_selector.py
    # Ensure LightGBM is installed. If gpu_enabled=True, ensure it's GPU-compiled LightGBM.
```

I've created the `src/features/feature_selector.py` file. It includes:
- `train_lgbm_and_get_rmse`: A helper function to train a LightGBM model and get its RMSE. It handles data type conversion to float32 for LightGBM and includes basic GPU configuration.
- `select_features_batchwise`: The main function for batch feature selection. It splits data (currently random split, but noted it should be time-series aware if applicable), iterates through feature batches, trains models, and selects batches based on RMSE improvement.
- An `if __name__ == '__main__':` block provides a basic example of how to use `select_features_batchwise` with dummy data.

**Important Considerations & Potential Issues in the current draft:**
- **Time Series Split:** The current data splitting uses `sklearn.model_selection.train_test_split` with `shuffle=False`. For time-series data, a proper chronological split is crucial. This might involve sorting by a 'Date' index/column and then taking the first N% for train and the rest for validation. This needs to be robustly handled in the actual pipeline.
- **cuML for Splitting:** If using cuDF, `cuml.model_selection.train_test_split` would be more efficient than converting to pandas for splitting and then converting back. However, this adds a dependency on cuML.
- **Initial Feature Set Handling:** If `initial_feature_set` is empty, the current logic will accept the first batch that provides any finite RMSE (as `float('inf')` is the starting `best_rmse`). This is generally okay. The added logic for `if i == 0 and not initial_feature_set:` and then `if not selected_features:` ensures the first batch is accepted if no initial features were provided to establish a baseline.
- **LightGBM Parameters:** The default LightGBM parameters in `train_lgbm_and_get_rmse` are set for relatively fast training. These can be overridden via `lgbm_params_override`.
- **Reproducibility:** Seeds are set for LightGBM, but shuffling of candidate features is commented out as `np.random.shuffle` doesn't directly work on a list of strings in a way that's easily GPU accelerated without extra steps. If feature order matters, this could be revisited.
- **RMSE Calculation for cuDF:** The RMSE calculation `np.sqrt(np.mean((y_val_lgb.to_numpy() ... - preds)**2))` converts `y_val_lgb` to a numpy array. If `y_val_lgb` and `preds` are cuDF/cuPy arrays, using `cp.sqrt(cp.mean(...))` would be more efficient and keep data on the GPU. I'll make this adjustment.

Next, I will modify the RMSE calculation to use `cupy` if inputs are cuDF/cuPy and then proceed to update the config file.Okay, I've created the `src/features/feature_selector.py` file with the main logic. Now, I'll refine the RMSE calculation to use `cupy` when appropriate and then proceed to update the config file.

First, let's adjust the `train_lgbm_and_get_rmse` function for `cupy` RMSE calculation.
