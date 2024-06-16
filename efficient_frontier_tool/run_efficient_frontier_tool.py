from config.config import efficient_frontier_config
from efficient_frontier_tool.efficient_frontier_helper import (pull_multiple_yahoo_ticker_data,
                                                               simulate_portfolio,
                                                               find_minimum_risk_portfolio,
                                                               find_highest_sharp_portfolio)
import matplotlib.pyplot as plt

list_ticker_name = efficient_frontier_config['tickers']

annualized_expected_return, annualized_asset_cov = pull_multiple_yahoo_ticker_data(list_ticker_name)

simulated_portfolio_return = simulate_portfolio(annualized_expected_return, annualized_asset_cov, nr_simulation=10000)

minimum_risk_portfolio = find_minimum_risk_portfolio(simulated_portfolio_return)

highest_sharp_portfolio = find_highest_sharp_portfolio(simulated_portfolio_return)

plt.show()