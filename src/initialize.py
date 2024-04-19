import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
import numerapi
from config.config import ROOT_DIR, config_dict, download_file


# configurate pyspark env variable
pyspark_env_var_path = f"--jars {ROOT_DIR.joinpath('rapids-4-spark_2.12-21.12.0.jar')}," \
                       f"{ROOT_DIR.joinpath('cudf-21.12.2-cuda11.jar')} --master local[*] pyspark-shell"
os.environ['PYSPARK_SUBMIT_ARGS'] = pyspark_env_var_path

# Initialize SparkSession
def initialization():
    # download required jar files
    for value in config_dict['setup_files'].values():
        download_file(value)

    spark = SparkSession.builder.appName("numerai").getOrCreate()

    # spark.sparkContext.addPyFile("path/to/your/additional.jar")
    # # Additional configuration for packages
    # spark.conf.set("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    # spark.conf.set("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.1")

    napi = numerapi.SignalsAPI()
    current_ds = napi.get_current_round()
    print(f'{len(config_dict["setup_files"])} files are downloaded.'
          f'The spark session is initiated.'
          f'Current round is round {current_ds}.')

    return spark, napi

