import requests
import os
import toml
from pathlib import Path

# Directory setup
ROOT_DIR = Path(os.getcwd())
INPUT_DIR = ROOT_DIR.joinpath('data', 'raw')
PROCESSED_DIR = ROOT_DIR.joinpath('data', 'processed')
OUTPUT_DIR = ROOT_DIR.joinpath('submission_output')

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def read_toml_config(file_name: str):
    file_path = ROOT_DIR.joinpath('config', file_name)
    try:
        with open(file_path, "r") as f:
            config_data = toml.load(f)
        return config_data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: Failed to read TOML config file. {e}")
        return None

def download_file(file_config_dict):
    response = requests.get(file_config_dict['url'])
    if response.status_code == 200:
        output_path = INPUT_DIR.joinpath(file_config_dict['output_file'])
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully to {output_path}")
    else:
        print(f"Failed to download the file from {file_config_dict['url']}")

# Read config
config_dict = read_toml_config("config.toml")

if config_dict:
    # Setup files
    for file_key, file_config in config_dict.get('setup_files', {}).items():
        download_file(file_config)

    # Run parameters
    RUN_PARAMS = config_dict.get('run_parameter', {})
    START_DATE = str(RUN_PARAMS.get('START', "2010-01-01"))
    END_DATE = str(RUN_PARAMS.get('END', "2023-12-31"))
    FETCH_VIA_API = RUN_PARAMS.get('FETCH_VIA_API', False)
    SEED = RUN_PARAMS.get('SEED', 42)
    DS_OVERRIDE = RUN_PARAMS.get('DS_OVERRIDE', False)
    CURRENT_DS_OVERRIDE = RUN_PARAMS.get('CURRENT_DS_OVERRIDE', None)

    # Other configurations
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
    if not FRED_API_KEY:
        print("Warning: FRED_API_KEY not set in environment variables. Using dummy data for economic indicators.")
        FRED_API_KEY = 'dummy_key'

    NUMERAI_DATASET = "signals/v1.0/train.parquet"

    # Ensure Numerai dataset directory exists
    NUMERAI_DIR = INPUT_DIR.joinpath(NUMERAI_DATASET).parent
    NUMERAI_DIR.mkdir(parents=True, exist_ok=True)

    # Model parameters
    LGBM_PARAMS = {
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 100
    }

    NN_PARAMS = {
        "layers": [64, 32, 16],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32
    }

    H2O_PARAMS = {
        "max_models": 20,
        "seed": SEED,
        "max_runtime_secs": 300
    }

    # Spark configuration
    SPARK_CONFIG = {
        "app_name": "FinancialModelingPipeline",
        "master": "local[*]",
        "packages": [
            "com.microsoft.azure:synapseml_2.12:1.0.5",
            "org.apache.hadoop:hadoop-aws:3.2.0"
        ],
        "repositories": [
            "https://mmlspark.azureedge.net/maven"
        ]
    }

    # RAPIDS configuration
    RAPIDS_CONFIG = {
        "use_gpu": True,
        "gpu_memory_fraction": 0.8
    }

else:
    print("Failed to load configuration. Using default values.")
    # Define default values here
    START_DATE = "2010-01-01"
    END_DATE = "2023-12-31"
    FETCH_VIA_API = False
    SEED = 42
    DS_OVERRIDE = False
    CURRENT_DS_OVERRIDE = None
    FRED_API_KEY = 'dummy_key'
    NUMERAI_DATASET = "signals/v1.0/train.parquet"
    LGBM_PARAMS = {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 100}
    NN_PARAMS = {"layers": [64, 32, 16], "dropout_rate": 0.2, "learning_rate": 0.001, "epochs": 100, "batch_size": 32}
    H2O_PARAMS = {"max_models": 20, "seed": SEED, "max_runtime_secs": 300}
    SPARK_CONFIG = {
        "app_name": "FinancialModelingPipeline",
        "master": "local[*]",
        "packages": ["com.microsoft.azure:synapseml_2.12:1.0.5", "org.apache.hadoop:hadoop-aws:3.2.0"],
        "repositories": ["https://mmlspark.azureedge.net/maven"]
    }
    RAPIDS_CONFIG = {"use_gpu": True, "gpu_memory_fraction": 0.8}

# Ensure NUMERAI_DIR exists
NUMERAI_DIR = INPUT_DIR.joinpath(NUMERAI_DATASET).parent
NUMERAI_DIR.mkdir(parents=True, exist_ok=True)