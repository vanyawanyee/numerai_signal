from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from h2o.automl import H2OAutoML
import h2o
import pandas as pd

def train_lgbm(X_train, y_train):
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_nn(X_train, y_train, input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    return model

def train_h2o_automl(X_train, y_train, X_test, y_test):
    h2o.init(strict_version_check=False)
    train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
    
    y = y_train.name
    x = train.columns
    x.remove(y)
    
    aml = H2OAutoML(max_models=20, seed=1, max_runtime_secs=300)
    aml.train(x=x, y=y, training_frame=train, leaderboard_frame=test)
    
    return aml.leader

def train_models(data):
    features = [col for col in data.columns if col not in ['DAX_LogReturn', 'Tesla_LogReturn']]
    X_dax, X_tesla = data[features], data[features]
    y_dax, y_tesla = data['DAX_LogReturn'], data['Tesla_LogReturn']
    
    X_train_dax, X_test_dax, y_train_dax, y_test_dax = train_test_split(X_dax, y_dax, test_size=0.2, random_state=42)
    X_train_tesla, X_test_tesla, y_train_tesla, y_test_tesla = train_test_split(X_tesla, y_tesla, test_size=0.2, random_state=42)
    
    scaler_dax = StandardScaler()
    scaler_tesla = StandardScaler()
    X_train_dax_scaled = scaler_dax.fit_transform(X_train_dax)
    X_test_dax_scaled = scaler_dax.transform(X_test_dax)
    X_train_tesla_scaled = scaler_tesla.fit_transform(X_train_tesla)
    X_test_tesla_scaled = scaler_tesla.transform(X_test_tesla)
    
    models = {
        'lgbm_dax': train_lgbm(X_train_dax_scaled, y_train_dax),
        'nn_dax': train_nn(X_train_dax_scaled, y_train_dax, X_train_dax_scaled.shape[1]),
        'h2o_dax': train_h2o_automl(X_train_dax, y_train_dax, X_test_dax, y_test_dax),
        'lgbm_tesla': train_lgbm(X_train_tesla_scaled, y_train_tesla),
        'nn_tesla': train_nn(X_train_tesla_scaled, y_train_tesla, X_train_tesla_scaled.shape[1]),
        'h2o_tesla': train_h2o_automl(X_train_tesla, y_train_tesla, X_test_tesla, y_test_tesla)
    }
    
    return models, (scaler_dax, scaler_tesla)