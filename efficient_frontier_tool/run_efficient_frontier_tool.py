from config.config import efficient_frontier_config
from efficient_frontier_tool.efficient_frontier_helper import pull_multiple_yahoo_ticker_data, simulate_portfolio
import matplotlib.pyplot as plt

list_ticker_name = efficient_frontier_config['tickers']

annualized_rate, annualized_asset_cov = pull_multiple_yahoo_ticker_data(list_ticker_name)

portfolio_result, portfolio_detail, sharp_ratio = simulate_portfolio(annualized_rate, annualized_asset_cov, nr_simulation=10000)

plt.show()