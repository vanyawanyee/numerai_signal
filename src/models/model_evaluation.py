"""
Model Evaluation Module
=======================

This module provides functions to evaluate the performance of trained machine learning
models. It calculates standard regression metrics like Root Mean Squared Error (RMSE)
and R-squared (R2).

The primary function, `evaluate_models`, takes a dictionary of trained models and
the corresponding test data (features and target) to generate predictions and compute
evaluation metrics. It also handles an ensemble prediction by averaging the outputs
of the individual models.

The `print_evaluation_results` function formats and displays these metrics in a
user-friendly way using the logging module.

Key Considerations for Evaluation:
- **Correct Test Set:** It is crucial that the data passed for evaluation (`X_test_scaled`,
  `y_test`, `X_test_unscaled_pd`) is the actual hold-out test set that was not seen
  by the models during training or hyperparameter tuning. The `train_models` function
  in `model_training.py` is responsible for creating this split, and these test sets
  should be passed to `main.py` and then to this evaluation module.
- **Data Types:** The function handles different data types (pandas, cuDF) for features
  and ensures compatibility with model prediction methods (e.g., H2O models expect H2OFrames).
- **Model Compatibility:** Predictions are made assuming standard `.predict()` methods.
  Output shapes (e.g., `.flatten()` for NN predictions) are handled as needed.
"""
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd # For H2OFrame conversion and type checking
import h2o # For H2O model predictions
import logging # For logging results
import cudf # For type checking

# Configure logger for this module
logger = logging.getLogger(__name__)

def evaluate_models(models, X_test_scaled, y_test, X_test_unscaled_pd):
    """
    Evaluates trained models on the test set and computes RMSE and R2 score.
    Also computes metrics for a simple ensemble (average of predictions).

    Args:
        models (dict): A dictionary of trained model objects. 
                       Keys are model names (e.g., 'lgbm_target_20d').
        X_test_scaled (pd.DataFrame or cudf.DataFrame): Scaled feature data for the test set.
                                                        Used by LightGBM and Neural Network.
        y_test (pd.Series or cudf.Series): Actual target values for the test set.
        X_test_unscaled_pd (pd.DataFrame): Unscaled feature data for the test set (pandas DataFrame).
                                           Used by H2O AutoML models.
                                           
    Returns:
        dict: A dictionary containing evaluation results (RMSE, R2) for each model
              and the ensemble. Structure: {'model_name': {'RMSE': value, 'R2': value, ...}}
    """
    logger.info(f"Starting model evaluation for {len(models)} models.")
    results = {}
    
    # Convert y_test to pandas Series or NumPy array for metric calculations if it's cuDF
    # This ensures compatibility with sklearn.metrics functions.
    if isinstance(y_test, cudf.Series):
        y_test_eval = y_test.to_pandas() 
    elif isinstance(y_test, pd.Series):
        y_test_eval = y_test
    else: # Assuming numpy array or list-like
        y_test_eval = np.asarray(y_test)

    model_predictions = {} # To store predictions for ensembling

    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        try:
            if model is None:
                logger.warning(f"Model '{model_name}' is None. Skipping evaluation.")
                results[model_name] = {'RMSE': np.nan, 'R2': np.nan}
                model_predictions[model_name] = np.full(len(y_test_eval), np.nan) # Placeholder for ensemble
                continue

            preds = None
            if 'h2o' in model_name.lower():
                # H2O models expect H2OFrame. X_test_unscaled_pd is used here.
                logger.debug(f"Predicting with H2O model. Input X_test_unscaled_pd shape: {X_test_unscaled_pd.shape}")
                if X_test_unscaled_pd.empty:
                     logger.warning(f"X_test_unscaled_pd is empty for H2O model {model_name}. Skipping prediction.")
                     preds = np.full(len(y_test_eval), np.nan)
                else:
                    h2o_test_frame = h2o.H2OFrame(X_test_unscaled_pd)
                    preds_h2o = model.predict(h2o_test_frame)
                    preds = preds_h2o.as_data_frame().iloc[:, 0].values # Assuming first column is prediction
            elif 'lgbm' in model_name.lower() or 'nn' in model_name.lower():
                # LightGBM and NN use scaled features.
                # Ensure X_test_scaled is in the format expected by the model (e.g., numpy for some NNs)
                # LGBM usually handles pandas/cudf/numpy. Keras predict usually handles numpy.
                logger.debug(f"Predicting with LGBM/NN model. Input X_test_scaled type: {type(X_test_scaled)}, shape: {X_test_scaled.shape if hasattr(X_test_scaled, 'shape') else 'N/A'}")
                
                # If X_test_scaled is cuDF and model is Keras NN, convert to numpy
                if 'nn' in model_name.lower() and isinstance(X_test_scaled, cudf.DataFrame):
                    X_test_for_pred = X_test_scaled.to_numpy()
                else:
                    X_test_for_pred = X_test_scaled
                
                preds = model.predict(X_test_for_pred)
                if hasattr(preds, 'flatten'): # Keras NN might return (n, 1) array
                    preds = preds.flatten()
            else:
                logger.warning(f"Unknown model type for '{model_name}'. Prediction logic might be missing. Attempting direct predict.")
                preds = model.predict(X_test_scaled) # Default attempt with scaled data

            if preds is None: # Should not happen if logic above is correct
                raise ValueError("Predictions are None, check model type handling.")

            # Convert cuDF/cupy predictions to numpy for sklearn metrics
            if hasattr(preds, 'to_numpy'): # For cuDF Series/DataFrame
                preds = preds.to_numpy()
            elif 'cupy' in str(type(preds)): # For CuPy arrays
                preds = preds.get()

            model_predictions[model_name] = preds
            rmse = np.sqrt(mean_squared_error(y_test_eval, preds))
            r2 = r2_score(y_test_eval, preds)
            results[model_name] = {'RMSE': rmse, 'R2': r2}
            logger.info(f"Model: {model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}", exc_info=True)
            results[model_name] = {'RMSE': np.nan, 'R2': np.nan}
            model_predictions[model_name] = np.full(len(y_test_eval), np.nan) # Placeholder for ensemble

    # Ensemble prediction (simple average)
    # Filter out any models that failed prediction (preds are NaN arrays)
    valid_preds_for_ensemble = [p for p in model_predictions.values() if not np.all(np.isnan(p))]
    if len(valid_preds_for_ensemble) > 0:
        # Ensure all prediction arrays have the same length as y_test_eval
        # This is important if some model predictions failed and were replaced by NaNs
        # or if a model predicted a different number of samples (should not happen with correct test set).
        # We only average predictions from models that successfully produced output of correct length.
        
        # Filter predictions that match y_test_eval length.
        # This implicitly handles cases where a model might have failed and preds are just `np.nan` (single value)
        # or if a model somehow returned an array of a different length.
        aligned_preds = [p for p in valid_preds_for_ensemble if len(p) == len(y_test_eval)]

        if len(aligned_preds) > 0:
            ensemble_pred = np.mean(aligned_preds, axis=0)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test_eval, ensemble_pred))
            ensemble_r2 = r2_score(y_test_eval, ensemble_pred)
            results['ensemble'] = {
                'RMSE': ensemble_rmse, 
                'R2': ensemble_r2,
                'y_test': y_test_eval,       # For plotting or further analysis
                'ensemble_pred': ensemble_pred 
            }
            logger.info(f"Ensemble (average of {len(aligned_preds)} models) - RMSE: {ensemble_rmse:.4f}, R2: {ensemble_r2:.4f}")
        else:
            logger.warning("No valid model predictions available for ensembling or length mismatch.")
            results['ensemble'] = {'RMSE': np.nan, 'R2': np.nan}
    else:
        logger.warning("No model predictions available for ensembling.")
        results['ensemble'] = {'RMSE': np.nan, 'R2': np.nan}
        
    logger.info("Model evaluation completed.")
    return results

def print_evaluation_results(results):
    """
    Prints the evaluation results (RMSE and R2) for each model.

    Args:
        results (dict): A dictionary of evaluation results from `evaluate_models`.
    """
    logger.info("\n--- Model Evaluation Results ---")
    for model_name, metrics in results.items():
        if 'RMSE' in metrics and 'R2' in metrics: # Check if essential metrics are present
            rmse = metrics['RMSE']
            r2 = metrics['R2']
            if pd.notna(rmse) and pd.notna(r2): # Check for NaN values before formatting
                logger.info(f"{model_name.replace('_', ' ').title()}:")
                logger.info(f"  RMSE: {rmse:.4f}")
                logger.info(f"  R2 Score: {r2:.4f}")
            else:
                logger.warning(f"{model_name.replace('_', ' ').title()}: Metrics are NaN (evaluation might have failed).")
        else:
            logger.warning(f"Metrics (RMSE/R2) not found for model '{model_name}'.")
    logger.info("--- End of Evaluation Results ---")