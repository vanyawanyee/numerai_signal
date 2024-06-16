import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.config import efficient_frontier_config


class YahooData():
    def __init__(self, ticker, start_date=None, end_date=None):
        self.ticker_name = ticker
        self.start_date = start_date
        self.end_date = end_date


    def fetch_dataframe(self):
        if self.start_date == None:
            self.dataframe = yf.download(self.ticker_name, period='1y')
        else:
            self.dataframe = yf.download(self.ticker_name, start=self.start_date, end=self.end_date)

        self.dataframe.set_index(self.dataframe.index.date, inplace=True)
        return self.dataframe


    def calculate_return_variance(self):
        self.std = np.std(self.dataframe['Adj Close'])
        self.var = np.var(self.dataframe['Adj Close'])
        self.dataframe.loc[:, 'change_rate'] = self.dataframe['Adj Close'].pct_change()
        self.dataframe.dropna(axis=0, inplace=True)
        # self.dataframe.loc[:, 'annualized_return_rate'] = (1 + self.dataframe['change_rate'])**(252/len(self.dataframe)) - 1
        # self.expected_return = np.mean(self.dataframe['annualized_return_rate'])



def generate_weight_across_asset(n:int):
    # asset_weight = np.ones(n)
    # simulated_weight = np.random.dirichlet(asset_weight)
    simulated_proportion = np.random.uniform(0,1,n)
    simulated_weight = np.round(simulated_proportion / simulated_proportion.sum(),3)
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
    return annualized, all_asset_dataframe.cov() * trading_day_per_year


def simulate_portfolio(annualized_expected_return, annualized_asset_cov, nr_simulation=5000):
    portfolio_return = pd.DataFrame()
    portfolio_return['port_std'] = np.nan
    portfolio_return['port_return'] = np.nan
    portfolio_return['sharp_ratio'] = np.nan
    for i in range(len(annualized_expected_return)):
        portfolio_return[f'{annualized_expected_return.index[i]}_weight'] = np.nan
        portfolio_return[f'{annualized_expected_return.index[i]}_return'] = np.nan
    risk_free_rate = efficient_frontier_config['risk_free_rate']
    for x in range(nr_simulation):
        new_row_dict = {}
        asset_weight = generate_weight_across_asset(len(annualized_expected_return))
        port_std = round(np.sqrt(np.dot(asset_weight, np.dot(asset_weight, annualized_asset_cov))),4)
        port_return = round(np.dot(asset_weight, annualized_expected_return),4)
        sharp_ratio = round((port_return - risk_free_rate) / port_std,4)
        new_row_dict['port_std'] = port_std
        new_row_dict['port_return'] = port_return
        new_row_dict['sharp_ratio'] = sharp_ratio
        for i in range(len(annualized_expected_return)):
            new_row_dict[f'{annualized_expected_return.index[i]}_weight'] = asset_weight[i]
            new_row_dict[f'{annualized_expected_return.index[i]}_return'] =  annualized_expected_return[i]
        portfolio_return = portfolio_return._append([new_row_dict], ignore_index=True)

    plt.scatter(
        [value * 100 for value in portfolio_return['port_std'].values],
        [value * 100 for value in portfolio_return['port_return'].values],
        c=portfolio_return['sharp_ratio'].values,
        cmap="viridis",
        alpha=0.75,
    )
    plt.xlabel("Risk (%)")
    plt.ylabel("Expected Returns (%)")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid()

    return portfolio_return


def find_minimum_risk_portfolio(simulated_portfolio_return):
    minimum_risk = simulated_portfolio_return['port_std'].min()
    minimum_risk_portfolio = simulated_portfolio_return[simulated_portfolio_return['port_std'] == minimum_risk]


    plt.scatter(
        minimum_risk_portfolio['port_std'].values[0] * 100,
        minimum_risk_portfolio['port_return'].values[0] * 100,
        marker='*',
        color='red',
        s=200,
        edgecolors='black',
        zorder=5
    )
    print(f'''
    The red one represents the portfolio with the minimum risk:
    return: {minimum_risk_portfolio['port_return'].values[0]}
    risk: {minimum_risk_portfolio['port_std'].values[0]}
    '''
          )


    return minimum_risk_portfolio


def find_highest_sharp_portfolio(simulated_portfolio_return):
    highest_sharp = simulated_portfolio_return['sharp_ratio'].max()
    highest_sharp_portfolio = simulated_portfolio_return[simulated_portfolio_return['sharp_ratio'] == highest_sharp]

    plt.scatter(
        highest_sharp_portfolio['port_std'].values[0] * 100,
        highest_sharp_portfolio['port_return'].values[0] * 100,
        marker='*',
        color='green',
        s=200,
        edgecolors='black',
        zorder=5
    )

    print(f'''
    The green start represents the portfolio with the highest sharp ratio:
    return: {highest_sharp_portfolio['port_return'].values[0]}
    risk: {highest_sharp_portfolio['port_std'].values[0]}
    '''
          )

    return highest_sharp_portfolio



