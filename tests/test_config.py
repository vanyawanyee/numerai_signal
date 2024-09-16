import pytest
from pathlib import Path
from config.config import (
    ROOT_DIR, INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR,
    read_toml_config, START_DATE, END_DATE, FETCH_VIA_API,
    SEED, DS_OVERRIDE, CURRENT_DS_OVERRIDE, FRED_API_KEY,
    NUMERAI_DATASET, LGBM_PARAMS, NN_PARAMS, H2O_PARAMS,
    SPARK_CONFIG, RAPIDS_CONFIG
)

def test_directory_structure():
    assert ROOT_DIR.exists(), "ROOT_DIR should exist"
    assert INPUT_DIR.exists(), "INPUT_DIR should exist"
    assert PROCESSED_DIR.exists(), "PROCESSED_DIR should exist"
    assert OUTPUT_DIR.exists(), "OUTPUT_DIR should exist"

def test_toml_config():
    config = read_toml_config("config.toml")
    assert config is not None, "Should be able to read config.toml"
    assert 'setup_files' in config, "config should have 'setup_files' section"
    assert 'run_parameter' in config, "config should have 'run_parameter' section"

def test_run_parameters():
    assert isinstance(START_DATE, str), "START_DATE should be a string"
    assert isinstance(END_DATE, str), "END_DATE should be a string"
    assert isinstance(FETCH_VIA_API, bool), "FETCH_VIA_API should be a boolean"
    assert isinstance(SEED, int), "SEED should be an integer"
    assert isinstance(DS_OVERRIDE, bool), "DS_OVERRIDE should be a boolean"
    assert CURRENT_DS_OVERRIDE is None or isinstance(CURRENT_DS_OVERRIDE, int), "CURRENT_DS_OVERRIDE should be None or an integer"

def test_api_keys():
    assert FRED_API_KEY != 'your_fred_api_key_here', "FRED_API_KEY should be set"

def test_dataset_paths():
    assert isinstance(NUMERAI_DATASET, str), "NUMERAI_DATASET should be a string"
    numerai_path = Path(INPUT_DIR, NUMERAI_DATASET)
    assert numerai_path.parent.exists(), f"Directory for {NUMERAI_DATASET} should exist"

def test_model_parameters():
    assert isinstance(LGBM_PARAMS, dict), "LGBM_PARAMS should be a dictionary"
    assert isinstance(NN_PARAMS, dict), "NN_PARAMS should be a dictionary"
    assert isinstance(H2O_PARAMS, dict), "H2O_PARAMS should be a dictionary"

def test_spark_config():
    assert isinstance(SPARK_CONFIG, dict), "SPARK_CONFIG should be a dictionary"
    assert 'app_name' in SPARK_CONFIG, "SPARK_CONFIG should have 'app_name'"
    assert 'master' in SPARK_CONFIG, "SPARK_CONFIG should have 'master'"
    assert 'packages' in SPARK_CONFIG, "SPARK_CONFIG should have 'packages'"

def test_rapids_config():
    assert isinstance(RAPIDS_CONFIG, dict), "RAPIDS_CONFIG should be a dictionary"
    assert 'use_gpu' in RAPIDS_CONFIG, "RAPIDS_CONFIG should have 'use_gpu'"
    assert isinstance(RAPIDS_CONFIG['use_gpu'], bool), "RAPIDS_CONFIG['use_gpu'] should be a boolean"