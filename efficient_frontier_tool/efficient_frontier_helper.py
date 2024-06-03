import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.config import efficient_frontier_config


class YahooData(yf.Ticker):
    def __init__(self, ticker, start_date=None):
        super().__init__(ticker)
        self.ticker_name = ticker
        self.start_date = start_date


    def fetch_dataframe(self):
        if self.start_date == None:
            self.dataframe = super().history(period='1y')
        else:
            self.dataframe = super().history(start=self.start_date)

        self.dataframe.set_index(self.dataframe.index.date, inplace=True)
        return self.dataframe


    def calculate_return_variance(self):
        self.std = np.std(self.dataframe['Close'])
        self.var = np.var(self.dataframe['Close'])
        self.dataframe.loc[:, 'change_rate'] = self.dataframe['Close'].pct_change()
        self.dataframe.dropna(axis=0, inplace=True)
        # self.dataframe.loc[:, 'annualized_return_rate'] = (1 + self.dataframe['change_rate'])**(252/len(self.dataframe)) - 1
        # self.expected_return = np.mean(self.dataframe['annualized_return_rate'])



def generate_weight_across_asset(n:int):
    asset_weight = np.ones(n)
    simulated_weight = np.random.dirichlet(asset_weight)
    return simulated_weight


def pull_multiple_yahoo_ticker_data(list_ticker_name):
    all_asset_dataframe = pd.DataFrame()
    for ticker_name in list_ticker_name:
        asset = YahooData(ticker_name)
        asset.fetch_dataframe()
        asset.calculate_return_variance()
        all_asset_dataframe.loc[:, ticker_name] = asset.dataframe['change_rate']

    trading_day_per_year = efficient_frontier_config['number_of_trading_days']
    # multiple cumulative daily return and power to the number of year 252 / dataframe records
    annualized = (1 + all_asset_dataframe).prod() ** (trading_day_per_year/len(all_asset_dataframe)) - 1
    return annualized, all_asset_dataframe.cov() * np.sqrt(trading_day_per_year)


def simulate_portfolio(annualized_rate, annualized_asset_cov, nr_simulation=5000):
    portfolio_result = {}
    portfolio_detail = {}
    risk_free_rate = efficient_frontier_config['risk_free_rate']
    sharp_ratio_list = []
    for x in range(nr_simulation):
        asset_weight = generate_weight_across_asset(len(annualized_rate))
        port_std = np.sqrt(np.dot(asset_weight, np.dot(asset_weight, annualized_asset_cov)))
        port_return = np.dot(asset_weight, annualized_rate)
        portfolio_result[port_return] = port_std
        portfolio_detail[port_return] = asset_weight
        sharp_ratio = (port_return - risk_free_rate) / port_std
        sharp_ratio_list.append(sharp_ratio)

    plt.scatter(
        [round(value * 100, 3) for value in portfolio_result.values()],
        [key * 100 for key in portfolio_result.keys()],
        c=sharp_ratio_list,
        cmap="viridis",
        alpha=0.75,
    )
    plt.xlabel("Risk (%)")
    plt.ylabel("Expected Returns (%)")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid()
    # plt.show()

    return portfolio_result, portfolio_detail, sharp_ratio_list



