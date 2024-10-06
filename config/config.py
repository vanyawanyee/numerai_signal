import requests
import os
import toml
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
INPUT_DIR = ROOT_DIR.joinpath('data')
OUTPUT_DIR = ROOT_DIR.joinpath('submission_output')

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
config_dict = read_toml_config("config.toml")
efficient_frontier_config = read_toml_config('efficient_frontier.toml')



