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
        "n_estimators": 100,
        "random_state": SEED, # Added for reproducibility
        # GPU specific parameters will be added by train_lgbm if RAPIDS_CONFIG['use_gpu'] is True
        # Example if you want to set them directly here (and modify train_lgbm to use them):
        # "device": "gpu",
        # "tree_learner": "data_parallel", 
        # "gpu_platform_id": 0, # Optional: typically auto-detected
        # "gpu_device_id": 0 # Optional: typically auto-detected or uses all available with data_parallel
                           # Forcing specific multiple IDs like '0,1,2' is not standard for LGBM single process.
    }

    NN_PARAMS = {
        "layers": [64, 32, 16], # Architecture of dense layers
        "dropout_rate": 0.2,    # Dropout rate after each dense layer
        "learning_rate": 0.001, # Optimizer learning rate
        "epochs": 100,          # Max number of training epochs
        "batch_size": 32,       # Per-replica batch size. Global batch size will be calculated by MirroredStrategy.
                                # E.g., if 3 GPUs, global batch size = 32 * 3 = 96.
        "optimizer": "adam",    # Optimizer choice (though Adam is hardcoded in train_nn for now)
        "enable_determinism": False # Added for TensorFlow op determinism
    }

    H2O_PARAMS = {
        "max_models": 20,       # Max number of models to train in AutoML
        "seed": SEED,           # For reproducibility
        "max_runtime_secs": 300, # Max time for AutoML run
        # "nfolds": 5,          # Optional: Number of folds for cross-validation. Can improve parallelism.
        # H2O discovers and uses available hardware (CPUs, GPUs for GPU-enabled algos like XGBoost) automatically.
        # `h2o.init(nthreads=-1)` in train_h2o_automl allows using all CPU cores.
        # Specific multi-GPU control for a single model is algorithm-dependent within H2O (e.g. XGBoost backend params).
        # AutoML will leverage this if the included algorithms are GPU-enabled.
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

    # Feature Selection Parameters
    FEATURE_SELECTION_PARAMS = config_dict.get('feature_selection_params', {})
    # Provide defaults if not in toml for critical params
    FEATURE_SELECTION_PARAMS.setdefault('INITIAL_FEATURE_SET', [])
    FEATURE_SELECTION_PARAMS.setdefault('FEATURE_BATCH_SIZE', 5000)
    FEATURE_SELECTION_PARAMS.setdefault('RMSE_IMPROVEMENT_THRESHOLD', 0.0001)
    FEATURE_SELECTION_PARAMS.setdefault('MAX_SELECTED_FEATURES', 10000)
    FEATURE_SELECTION_PARAMS.setdefault('TARGET_COLUMN_NAME', 'target_20d') # Default Numerai target
    FEATURE_SELECTION_PARAMS.setdefault('OVERALL_RMSE_TARGET', 0.16) # Added RMSE target
    FEATURE_SELECTION_PARAMS.setdefault('LGBM_SELECTION_CONFIG', { # Default LGBM params for selection
        'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'min_child_samples': 20, 'verbose': -1, 'n_jobs': -1, 'seed': SEED,
        # Add device: gpu for selection if desired and selector function handles it
    })

    # Submission Parameters
    # Ensure 'submission_params' key exists in config_dict before trying to get from it
    submission_params_from_toml = config_dict.get('submission_params', {})
    SUBMISSION_PARAMS = {
        "SUBMISSION_FILENAME_PREFIX": submission_params_from_toml.get("SUBMISSION_FILENAME_PREFIX", "numerai_signals_submission"),
        "GENERATE_SUBMISSION_FILE": submission_params_from_toml.get("GENERATE_SUBMISSION_FILE", True),
        # Default model key construction needs FEATURE_SELECTION_PARAMS to be defined first
        "MODEL_FOR_SUBMISSION": submission_params_from_toml.get("MODEL_FOR_SUBMISSION", f"lgbm_{FEATURE_SELECTION_PARAMS['TARGET_COLUMN_NAME']}"),
        "LIVE_DATA_FILENAME": submission_params_from_toml.get("LIVE_DATA_FILENAME", "live.parquet")
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
    LGBM_PARAMS = {
        "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 100, "random_state": SEED
    }
    NN_PARAMS = {
        "layers": [64, 32, 16], "dropout_rate": 0.2, "learning_rate": 0.001, 
        "epochs": 100, "batch_size": 32, "optimizer": "adam",
        "enable_determinism": False # Added for TensorFlow op determinism
    }
    H2O_PARAMS = {
        "max_models": 20, "seed": SEED, "max_runtime_secs": 300
    }
    SPARK_CONFIG = {
        "app_name": "FinancialModelingPipeline",
        "master": "local[*]",
        "packages": ["com.microsoft.azure:synapseml_2.12:1.0.5", "org.apache.hadoop:hadoop-aws:3.2.0"],
        "repositories": ["https://mmlspark.azureedge.net/maven"]
    }
    RAPIDS_CONFIG = {"use_gpu": True, "gpu_memory_fraction": 0.8}
    FEATURE_SELECTION_PARAMS = { # Duplicated defaults for the else: block
        'INITIAL_FEATURE_SET': [],
        'FEATURE_BATCH_SIZE': 5000,
        'RMSE_IMPROVEMENT_THRESHOLD': 0.0001,
        'MAX_SELECTED_FEATURES': 10000,
        'TARGET_COLUMN_NAME': 'target_20d', # Changed default to Numerai target
        'OVERALL_RMSE_TARGET': 0.16, # Added RMSE target
        'LGBM_SELECTION_CONFIG': {
            'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
            'min_child_samples': 20, 'verbose': -1, 'n_jobs': -1, 'seed': SEED
        }
    }
    # Define SUBMISSION_PARAMS in the else block as well
    SUBMISSION_PARAMS = {
        "SUBMISSION_FILENAME_PREFIX": "numerai_signals_submission",
        "GENERATE_SUBMISSION_FILE": True,
        "MODEL_FOR_SUBMISSION": f"lgbm_{FEATURE_SELECTION_PARAMS['TARGET_COLUMN_NAME']}", # Uses FEATURE_SELECTION_PARAMS defined above
        "LIVE_DATA_FILENAME": "live.parquet"
    }

# Ensure NUMERAI_DIR exists
NUMERAI_DIR = INPUT_DIR.joinpath(NUMERAI_DATASET).parent
NUMERAI_DIR.mkdir(parents=True, exist_ok=True)