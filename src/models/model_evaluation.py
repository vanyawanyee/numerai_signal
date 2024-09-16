from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import h2o

def evaluate_models(models, data, scalers):
    features = [col for col in data.columns if col not in ['DAX_LogReturn', 'Tesla_LogReturn']]
    X_dax, X_tesla = data[features], data[features]
    y_dax, y_tesla = data['DAX_LogReturn'], data['Tesla_LogReturn']
    
    _, X_test_dax, _, y_test_dax = train_test_split(X_dax, y_dax, test_size=0.2, random_state=42)
    _, X_test_tesla, _, y_test_tesla = train_test_split(X_tesla, y_tesla, test_size=0.2, random_state=42)
    
    scaler_dax, scaler_tesla = scalers
    X_test_dax_scaled = scaler_dax.transform(X_test_dax)
    X_test_tesla_scaled = scaler_tesla.transform(X_test_tesla)
    
    results = {}
    for market in ['dax', 'tesla']:
        X_test = X_test_dax_scaled if market == 'dax' else X_test_tesla_scaled
        y_test = y_test_dax if market == 'dax' else y_test_tesla
        
        lgbm_pred = models[f'lgbm_{market}'].predict(X_test)
        nn_pred = models[f'nn_{market}'].predict(X_test).flatten()
        h2o_pred = models[f'h2o_{market}'].predict(h2o.H2OFrame(X_test)).as_data_frame().values.flatten()
        
        ensemble_pred = (lgbm_pred + nn_pred + h2o_pred) / 3
        
        results[market] = {
            'lgbm_rmse': np.sqrt(mean_squared_error(y_test, lgbm_pred)),
            'nn_rmse': np.sqrt(mean_squared_error(y_test, nn_pred)),
            'h2o_rmse': np.sqrt(mean_squared_error(y_test, h2o_pred)),
            'ensemble_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'lgbm_r2': r2_score(y_test, lgbm_pred),
            'nn_r2': r2_score(y_test, nn_pred),
            'h2o_r2': r2_score(y_test, h2o_pred),
            'ensemble_r2': r2_score(y_test, ensemble_pred),
            'y_test': y_test,
            'ensemble_pred': ensemble_pred
        }
    
    return results

def print_evaluation_results(results):
    for market in results:
        print(f"\nResults for {market.upper()}:")
        print(f"LGBM RMSE: {results[market]['lgbm_rmse']:.4f}, R2: {results[market]['lgbm_r2']:.4f}")
        print(f"Neural Network RMSE: {results[market]['nn_rmse']:.4f}, R2: {results[market]['nn_r2']:.4f}")
        print(f"H2O AutoML RMSE: {results[market]['h2o_rmse']:.4f}, R2: {results[market]['h2o_r2']:.4f}")
        print(f"Ensemble RMSE: {results[market]['ensemble_rmse']:.4f}, R2: {results[market]['ensemble_r2']:.4f}")