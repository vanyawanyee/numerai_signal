import requests
import os
import toml
from pathlib import Path

ROOT_DIR = Path(os.getcwd())

def read_toml_config(file_name:str):
    file_path = ROOT_DIR.joinpath('config/' + file_name)
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
        with open(file_config_dict['output_file'], 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")


# read config
config = read_toml_config("config.toml")
if config:
    print(config)

# download required jar files
for value in config['setup_files'].values():
    download_file(value)

# configurate pyspark env variable
pyspark_env_var_path = f"--jars {ROOT_DIR.joinpath('rapids-4-spark_2.12-21.12.0.jar')}," \
                       f"{ROOT_DIR.joinpath('cudf-21.12.2-cuda11.jar')} --master local[*] pyspark-shell"
os.environ['PYSPARK_SUBMIT_ARGS'] = pyspark_env_var_path
