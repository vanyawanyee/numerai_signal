import sys
sys.path.insert(0, '.')
from pyspark.sql import SparkSession
from src.features.pandas_features import add_ewma
from src.features.pyspark_features import add_feats, add_advanced_feats
from src.features.build_poly_features import build_poly_features
from config import CFG, FeatureCFG

def build_features(spark: SparkSession, input_data: str) -> None:
    df = spark.read.parquet(input_data)
    df = add_feats(df)
    df = add_advanced_feats(df)
    df = df.withColumn('ewma', add_ewma(df['Close']))
    df = build_poly_features(df, FeatureCFG.POLY_FEATURE_COLS)
    df.write.mode("overwrite").parquet(f"{CFG.OUTPUT_DIR}/features.parquet")

