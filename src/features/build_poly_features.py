from pyspark.ml.feature import PolynomialExpansion, VectorAssembler
from config import FeatureCFG

def build_poly_features(df):
    existing_cols = df.columns
    input_cols = [col for col in FeatureCFG.POLY_FEATURE_COLS if col in existing_cols]

    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    df = assembler.transform(df)
    polyExpansion = PolynomialExpansion(degree=2, inputCol="features", outputCol="polyFeatures")
    df = polyExpansion.transform(df)
    df = df.drop('features')
    
    return df
