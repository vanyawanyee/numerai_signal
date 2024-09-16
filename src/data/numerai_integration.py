from numerapi import NumerAPI
import pandas as pd

def download_numerai_data():
    napi = NumerAPI()
    
    # List available datasets
    datasets = [f for f in napi.list_datasets() if f.startswith("signals/v1.0")]
    print("Available Numerai datasets:")
    for dataset in datasets:
        print(f"- {dataset}")
    
    # Download the training data
    train_file = "signals/v1.0/train.parquet"
    napi.download_dataset(train_file)
    
    # Load the downloaded data
    df = pd.read_parquet(train_file)
    
    print(f"Numerai data downloaded and loaded. Shape: {df.shape}")
    return df

def process_numerai_data(df):
    # data processing steps
    # - Handle missing values
    # - Feature engineering
    # - Feature selection
    
    print("Numerai data processed.")
    return df