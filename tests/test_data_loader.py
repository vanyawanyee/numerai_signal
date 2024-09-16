import pytest
import pandas as pd
from src.data.data_loader import load_market_data, load_economic_indicators
from config.config import START_DATE, END_DATE, FRED_API_KEY

def test_load_market_data():
    data = load_market_data(start_date=START_DATE, end_date=END_DATE)
    assert isinstance(data, pd.DataFrame), "load_market_data should return a DataFrame"
    assert not data.empty, "Market data should not be empty"
    assert 'DAX_Close' in data.columns, "DAX_Close should be in the columns"
    assert 'Tesla_Close' in data.columns, "Tesla_Close should be in the columns"

def test_load_economic_indicators():
    data = load_economic_indicators(api_key=FRED_API_KEY)
    assert isinstance(data, pd.DataFrame), "load_economic_indicators should return a DataFrame"
    assert not data.empty, "Economic indicators data should not be empty"
    assert 'Unemployment Rate' in data.columns, "Unemployment Rate should be in the columns"

def test_date_range():
    data = load_market_data(start_date=START_DATE, end_date=END_DATE)
    assert data.index.min().strftime('%Y-%m-%d') >= START_DATE, "Data should not start before START_DATE"
    assert data.index.max().strftime('%Y-%m-%d') <= END_DATE, "Data should not end after END_DATE"

@pytest.mark.parametrize("invalid_date", ["2000-13-01", "2000-01-32", "invalid_date"])
def test_invalid_date_input(invalid_date):
    with pytest.raises(ValueError):
        load_market_data(start_date=invalid_date, end_date=END_DATE)

def test_invalid_api_key():
    with pytest.raises(Exception):
        load_economic_indicators(api_key="invalid_key")