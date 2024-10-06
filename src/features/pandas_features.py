from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType
from config import FeatureCFG
import pandas as pd

@pandas_udf(returnType=DoubleType(), functionType=PandasUDFType.SCALAR)
def add_ewma(value: pd.Series) -> pd.Series:
    for span in FeatureCFG.EWMA_SPANS:
        ewma = value.ewm(span=span).mean()
    return ewma