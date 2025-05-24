"""
Model Training Module
=====================

This module is responsible for training various machine learning models, including
LightGBM, Neural Networks (TensorFlow/Keras), and H2O AutoML. It is designed to
leverage GPU acceleration where possible and provides functionalities for multi-GPU
training on a single node.

Key Features:
- Functions for training individual model types (`train_lgbm`, `train_nn`, `train_h2o_automl`).
- GPU acceleration support:
    - LightGBM: Uses `device='gpu'` and `tree_learner='data_parallel'`.
    - TensorFlow/Keras: Employs `tf.distribute.MirroredStrategy` for multi-GPU.
    - H2O: Relies on H2O's internal GPU discovery for compatible algorithms (e.g., XGBoost).
- Integration with global configuration (`config.config`) for model parameters,
  GPU settings, and random seeds.
- A main wrapper function `train_models` that orchestrates the training of all
  specified models for a given target variable, using selected features. This includes
  data splitting (train/test) and feature scaling.

The module aims to provide a standardized way to train different types of models
within the project, ensuring reproducibility and efficient use of hardware resources.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from h2o.automl import H2OAutoML
import h2o
import pandas as pd
import numpy as np 
import logging # For logging
import cudf # For type checking if RAPIDS is used

# Import global configurations
from config.config import LGBM_PARAMS, NN_PARAMS, H2O_PARAMS, RAPIDS_CONFIG, SEED 

# Configure logger for this module
logger = logging.getLogger(__name__)

def train_lgbm(X_train, y_train, lgbm_custom_params=None):
    """
    Trains a LightGBM Regressor model.

    Args:
        X_train (pd.DataFrame or cudf.DataFrame): Training feature data.
        y_train (pd.Series or cudf.Series): Training target data.
        lgbm_custom_params (dict, optional): Custom LightGBM parameters to override/add to global config.

    Returns:
        LGBMRegressor: Trained LightGBM model.
    """
    params = LGBM_PARAMS.copy()
    params['random_state'] = SEED # Ensure reproducibility

    if RAPIDS_CONFIG.get('use_gpu', False):
        params['device'] = 'gpu'
        params['tree_learner'] = 'data_parallel' # For multi-GPU on a single node
        # Note: LightGBM with 'data_parallel' typically uses all visible GPUs.
        # Explicitly setting 'num_gpu' or 'gpu_device_id' for multiple specific devices
        # in a single process is less common and depends on the LightGBM build and version.
        logger.info(f"Training LightGBM on GPU with data parallelism. Effective params: {params}")
    else:
        logger.info(f"Training LightGBM on CPU. Effective params: {params}")

    if lgbm_custom_params: 
        params.update(lgbm_custom_params)
        logger.info(f"Applied custom LGBM parameters: {lgbm_custom_params}")

    model = LGBMRegressor(**params)
    
    # Ensure data is float32, especially for GPU training
    if hasattr(X_train, 'astype'): X_train = X_train.astype(np.float32, errors='ignore')
    if hasattr(y_train, 'astype'): y_train = y_train.astype(np.float32, errors='ignore')
        
    try:
        model.fit(X_train, y_train)
        logger.info("LightGBM model training completed.")
    except Exception as e:
        logger.error(f"Error during LightGBM model fitting: {e}")
        # Log shapes and types for debugging
        logger.error(f"X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
        logger.error(f"y_train type: {type(y_train)}, shape: {y_train.shape if hasattr(y_train, 'shape') else 'N/A'}")
        raise
    return model

def train_nn(X_train, y_train, input_dim, nn_custom_params=None):
    """
    Trains a Neural Network model using TensorFlow/Keras with MirroredStrategy for multi-GPU.

    Args:
        X_train (pd.DataFrame, cudf.DataFrame, or np.ndarray): Training feature data.
        y_train (pd.Series, cudf.Series, or np.ndarray): Training target data.
        input_dim (int): Input dimension for the first Dense layer (number of features).
        nn_custom_params (dict, optional): Custom NN parameters to override/add to global config.

    Returns:
        tensorflow.keras.models.Sequential: Trained Keras Sequential model.
    """
    logger.info("Initializing TensorFlow MirroredStrategy for Neural Network training.")
    
    # Get current NN parameters and check for determinism flag
    current_nn_params = NN_PARAMS.copy()
    if nn_custom_params:
        current_nn_params.update(nn_custom_params)
        logger.info(f"Applied custom NN parameters: {nn_custom_params}")

    if current_nn_params.get('enable_determinism', False):
        try:
            tf.config.experimental.enable_op_determinism()
            logger.info("TensorFlow op determinism enabled via tf.config.experimental.enable_op_determinism().")
        except AttributeError:
            logger.warning("tf.config.experimental.enable_op_determinism() not found. Might be using an older TensorFlow version.")
        except Exception as e:
            logger.error(f"Error enabling TensorFlow op determinism: {e}")
            
    strategy = None
    try:
        physical_gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"Found {len(physical_gpus)} physical GPUs.")
        
        # Attempt to use up to 3 GPUs if available
        gpus_to_use_count = min(len(physical_gpus), 3) 
        
        if gpus_to_use_count > 0:
            gpu_devices_for_strategy = [f"/gpu:{i}" for i in range(gpus_to_use_count)] 
            logger.info(f"Attempting to use MirroredStrategy with devices: {gpu_devices_for_strategy}")
            strategy = tf.distribute.MirroredStrategy(devices=gpu_devices_for_strategy)
        else: 
            logger.info("No GPUs found or specified for MirroredStrategy. Using default strategy (CPU).")
            strategy = tf.distribute.get_strategy() 
    except RuntimeError as e: # Catch errors during strategy initialization (e.g., GPUs not usable)
        logger.error(f"Error initializing MirroredStrategy: {e}. Falling back to default strategy.")
        strategy = tf.distribute.get_strategy()
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during GPU strategy setup: {e}. Falling back to default strategy.")
        strategy = tf.distribute.get_strategy()

    logger.info(f"Using strategy: {type(strategy).__name__} with {strategy.num_replicas_in_sync} replica(s).")

    with strategy.scope():
        # Model definition
        # current_nn_params was already defined above and includes overrides

        model = Sequential()
        # Input layer: Dense layer with specified units, relu activation, and input dimension
        model.add(Dense(current_nn_params['layers'][0], activation='relu', input_dim=input_dim))
        model.add(Dropout(current_nn_params['dropout_rate'])) # Dropout for regularization
        
        # Hidden layers: Iterate through specified layer units
        for units in current_nn_params['layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(current_nn_params['dropout_rate']))
        
        # Output layer: Single neuron for regression output (no activation for linear output)
        model.add(Dense(1)) 
        
        # Compile the model with Adam optimizer and Mean Squared Error loss
        optimizer_choice = current_nn_params.get('optimizer', 'adam').lower()
        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=current_nn_params['learning_rate'])
        else: # Add other optimizers if needed, e.g. SGD
            logger.warning(f"Optimizer '{optimizer_choice}' not explicitly configured, defaulting to Adam.")
            optimizer = Adam(learning_rate=current_nn_params['learning_rate'])
            
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        logger.info("Neural Network model compiled successfully within strategy scope.")
        model.summary(print_fn=logger.info)


    # Global batch size for distributed training: per-replica batch size * number of replicas
    global_batch_size = current_nn_params['batch_size'] * strategy.num_replicas_in_sync
    logger.info(f"Global batch size for NN training: {global_batch_size} ({current_nn_params['batch_size']} per replica * {strategy.num_replicas_in_sync} replicas)")

    # Early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    # Ensure data is float32, common for NNs
    if hasattr(X_train, 'astype') and not isinstance(X_train, tf.data.Dataset): # tf.data.Dataset handles types internally
        X_train = X_train.astype(np.float32, errors='ignore')
    if hasattr(y_train, 'astype') and not isinstance(y_train, tf.data.Dataset):
        y_train = y_train.astype(np.float32, errors='ignore')

    logger.info(f"Starting Neural Network training for {current_nn_params['epochs']} epochs...")
    try:
        model.fit(X_train, y_train, 
                  epochs=current_nn_params['epochs'], 
                  batch_size=global_batch_size, 
                  validation_split=0.2, # Using a fraction of training data for validation during fit
                  callbacks=[early_stopping], 
                  verbose=1) 
        logger.info("Neural Network training completed.")
    except Exception as e:
        logger.error(f"Error during Neural Network model fitting: {e}")
        logger.error(f"X_train type: {type(X_train)}, y_train type: {type(y_train)}")
        if hasattr(X_train, 'shape'): logger.error(f"X_train shape: {X_train.shape}")
        if hasattr(y_train, 'shape'): logger.error(f"y_train shape: {y_train.shape}")
        raise
    return model

def train_h2o_automl(X_train, y_train, X_test, y_test, h2o_custom_params=None):
    """
    Trains an H2O AutoML model.

    H2O automatically discovers and utilizes available hardware resources (CPUs, GPUs).
    For multi-GPU, H2O's GPU-enabled algorithms (like XGBoost) can leverage them.
    This function initializes H2O to use all available CPU threads and relies on its
    internal mechanisms for GPU utilization.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        X_test (pd.DataFrame): Test feature data (for leaderboard).
        y_test (pd.Series): Test target data (for leaderboard).
        h2o_custom_params (dict, optional): Custom H2O AutoML parameters.

    Returns:
        H2OEstimator: The leader model from the H2O AutoML run.
    """
    logger.info("Initializing H2O cluster for AutoML training...")
    # Initialize H2O to use all available CPU cores (nthreads=-1).
    # H2O's GPU utilization is typically automatic for compatible algorithms (e.g., XGBoost).
    # `log_level="WARN"` reduces verbosity.
    try:
        h2o.init(nthreads=-1, strict_version_check=False, log_level="WARN") 
    except Exception as e:
        logger.error(f"Failed to initialize H2O cluster: {e}. H2O AutoML training will be skipped.")
        return None # Or raise an error if H2O is critical
    
    # Ensure input data are pandas DataFrames/Series as H2OFrame conversion is most robust from pandas.
    # If inputs are cuDF, they would have been converted to pandas in `train_models` before this call.
    if not isinstance(X_train, pd.DataFrame): X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series): 
        y_train = pd.Series(y_train, name='target') # Ensure y_train has a name for H2O
        logger.debug(f"Converted y_train to pandas Series with name '{y_train.name}'")
    if not isinstance(X_test, pd.DataFrame): X_test = pd.DataFrame(X_test)
    if not isinstance(y_test, pd.Series): 
        y_test = pd.Series(y_test, name=y_train.name) # Use same name as y_train
        logger.debug(f"Converted y_test to pandas Series with name '{y_test.name}'")

    # Ensure y_train and y_test have the same name.
    target_name = y_train.name if y_train.name else 'target' # Default if somehow still unnamed
    if y_train.name != target_name: y_train = y_train.rename(target_name)
    if y_test.name != target_name: y_test = y_test.rename(target_name)
    
    logger.info(f"Preparing H2OFrames for target '{target_name}'. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    try:
        train_h2o = h2o.H2OFrame(train_df)
        test_h2o = h2o.H2OFrame(test_df)
    except Exception as e:
        logger.error(f"Error converting pandas DataFrames to H2OFrames: {e}")
        return None
    
    # Define response (y) and predictor (x) columns for H2O
    y_col = target_name
    x_cols = list(train_h2o.columns)
    x_cols.remove(y_col) # All columns except the target are predictors
    
    # Prepare H2O AutoML parameters
    current_h2o_params = H2O_PARAMS.copy()
    current_h2o_params['seed'] = SEED # Ensure reproducibility
    if h2o_custom_params:
        current_h2o_params.update(h2o_custom_params)
        logger.info(f"Applied custom H2O AutoML parameters: {h2o_custom_params}")

    # Note on H2O Multi-GPU: H2O AutoML will use GPUs if available for algorithms like XGBoost.
    # It doesn't require explicit GPU ID list for `H2OAutoML` object itself.
    # `h2o.init(nthreads=-1)` is important for CPU parallelism which complements GPU tasks.
    # Forcing specific multi-GPU strategies (e.g., specific N GPUs for one model) is complex and
    # usually handled by H2O's internal scheduling or by running multiple H2O jobs.
    
    logger.info(f"Starting H2O AutoML with parameters: {current_h2o_params}")
    aml = H2OAutoML(**current_h2o_params)
    try:
        aml.train(x=x_cols, y=y_col, training_frame=train_h2o, leaderboard_frame=test_h2o)
        logger.info("H2O AutoML training complete.")
        logger.info("H2O AutoML Leader model:")
        logger.info(str(aml.leader)) # Log the leader model details
    except Exception as e:
        logger.error(f"Error during H2O AutoML training: {e}")
        return None # Or return the aml object if partial results are useful
    
    # Optional: Consider when/if to shutdown H2O cluster. 
    # If train_h2o_automl is called multiple times in a loop for different targets,
    # initializing and shutting down H2O each time can be slow.
    # A global H2O context managed in main.py might be better in such cases.
    # For now, assume it's managed elsewhere or this is a single call.
    # Example: h2o.cluster().shutdown(prompt=False)
    
    return aml.leader


# Modified train_models to accept selected features and target name
def train_models(all_features_df, selected_feature_names, target_column_name):
    """
    Main wrapper function to train multiple model types (LightGBM, NN, H2O AutoML).

    This function handles:
    1. Data preparation: Selecting features, splitting into train/test sets.
    2. Feature scaling: Using StandardScaler, applied to data for LightGBM and NN.
    3. Model training: Calls individual training functions for each model type.
    4. Returns trained models and the scaler object.

    Args:
        all_features_df (pd.DataFrame or cudf.DataFrame): DataFrame containing all original columns
                                                           (IDs like 'ticker', 'friday_date', 
                                                           all engineered features, and the target column).
        selected_feature_names (list): List of feature names to be used for training.
        target_column_name (str): Name of the column to be used as the target variable.

    Returns:
        tuple: (dict_of_models, dict_of_scalers)
               - dict_of_models: A dictionary where keys are model names (e.g., 'lgbm_target_20d')
                                 and values are the trained model objects.
               - dict_of_scalers: A dictionary where keys are target column names and values are
                                  the corresponding fitted StandardScaler objects.
    """
    logger.info(f"Starting model training process for target: '{target_column_name}' with {len(selected_feature_names)} features.")
    
    if not isinstance(target_column_name, str):
        logger.error("target_column_name must be a string.")
        raise ValueError("target_column_name must be a string.")

    if not selected_feature_names:
        logger.error("selected_feature_names list cannot be empty.")
        raise ValueError("selected_feature_names list cannot be empty.")
        
    # Prepare X (features) and y (target) using only selected features
    try:
        X = all_features_df[selected_feature_names]
        y = all_features_df[target_column_name]
    except KeyError as e:
        logger.error(f"KeyError when selecting features or target: {e}. Ensure selected_feature_names and target_column_name exist in all_features_df.")
        raise
    logger.debug(f"Prepared X with shape {X.shape} and y with shape {y.shape} for training.")

    # Data Splitting:
    # Convert to pandas for scikit-learn's train_test_split, ensuring consistency.
    # A time-series aware split should be used if data has chronological order.
    # Current split is random if shuffle=True (default for train_test_split).
    # For consistency with previous steps (like feature selection), ensure shuffle behavior is aligned.
    
    is_cudf_input = isinstance(X, cudf.DataFrame) 
    
    X_pd = X.to_pandas() if is_cudf_input else X.copy() # Use .copy() for pandas to avoid SettingWithCopyWarning on slices
    y_pd = y.to_pandas() if is_cudf_input else y.copy()

    logger.debug(f"Performing train/test split. Test size: 0.2, Random state: {SEED}, Shuffle: True.")
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
        X_pd, y_pd, test_size=0.2, random_state=SEED, shuffle=True 
        # Shuffle=True is common for IID data; for time series, this should be False and data pre-sorted.
        # The problem context implies general applicability, so shuffle=True might be a default.
        # However, for financial time series, this is usually False.
        # Re-confirming: The original code's train_test_split uses default shuffle=True.
    )
    logger.info(f"Data split: X_train_pd shape: {X_train_pd.shape}, X_test_pd shape: {X_test_pd.shape}")

    # Feature Scaling:
    # StandardScaler is fit on training data and used to transform both train and test sets.
    # It expects NumPy arrays or pandas DataFrames.
    logger.debug("Applying StandardScaler to features.")
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train_pd) # Fit on training data only
    X_test_scaled_np = scaler.transform(X_test_pd)   # Transform test data
    logger.debug("Feature scaling complete.")

    # Convert scaled data back to cuDF if GPU is enabled and original input was cuDF
    if is_cudf_input and RAPIDS_CONFIG.get('use_gpu', False):
        logger.debug("Converting scaled NumPy arrays back to cuDF DataFrames/Series.")
        X_train_scaled = cudf.DataFrame(X_train_scaled_np, columns=selected_feature_names, index=X_train_pd.index)
        y_train = cudf.Series(y_train_pd.values, index=y_train_pd.index, name=target_column_name) # Preserve index and name
        # X_test_scaled and y_test are also needed if evaluate_models expects them in cuDF format.
        # For now, H2O uses pandas versions, so this conversion is mainly for LGBM/NN if they benefit.
    else:
        X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=selected_feature_names, index=X_train_pd.index)
        y_train = y_train_pd
        # X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=selected_feature_names, index=X_test_pd.index)
        # y_test = y_test_pd
    logger.debug(f"X_train_scaled type: {type(X_train_scaled)}, y_train type: {type(y_train)}")


    # Train models:
    # LGBM and NN use scaled data.
    # H2O AutoML typically handles scaling internally or works well with unscaled data;
    # it's generally recommended to provide unscaled data to H2O unless specific reasons dictate otherwise.
    
    input_dim_nn = X_train_scaled.shape[1] # Number of features for NN input layer
    models = {}
    
    logger.info(f"\n--- Training LGBM for target: {target_column_name} ---")
    models[f'lgbm_{target_column_name}'] = train_lgbm(X_train_scaled, y_train)
    
    logger.info(f"\n--- Training NN for target: {target_column_name} ---")
    models[f'nn_{target_column_name}'] = train_nn(X_train_scaled, y_train, input_dim_nn)
    
    logger.info(f"\n--- Training H2O AutoML for target: {target_column_name} ---")
    # H2O uses the original pandas DataFrames (unscaled by this script's StandardScaler)
    # X_train_pd, y_train_pd, X_test_pd, y_test_pd are used here.
    h2o_leader_model = train_h2o_automl(X_train_pd, y_train_pd, X_test_pd, y_test_pd)
    if h2o_leader_model:
        models[f'h2o_{target_column_name}'] = h2o_leader_model
    else:
        logger.warning(f"H2O AutoML training failed or returned no leader model for target {target_column_name}.")
    
    # Store the scaler used for this target, as it's needed for inverse transforming predictions or scaling live data.
    scalers = {target_column_name: scaler} 
    
    logger.info(f"Model training process completed for target: '{target_column_name}'. Models trained: {list(models.keys())}")
    return models, scalers