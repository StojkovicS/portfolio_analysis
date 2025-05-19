### Preparatory work


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks, xlabel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Setting the plotting style to be colorblind-friendly
plt.style.use("seaborn-v0_8-colorblind")

# Loading data
stock_prices_df = pd.read_csv("faang_stocks.csv", index_col="Date")

# Changing the index to a datetime type allows for easier filtering and plotting.
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)
stock_prices_df


####################################################################################################################################################################################
# Plotting the stock prices
stock_prices_df.plot(
    title="FAANG stock prices from years 2020-2023",
    ylabel="Price (USD)"
)







############################## the expected annualized returns ##############################

# Daily returns per stock
daily_returns = stock_prices_df.pct_change().dropna()

# Equal weights to stocks in the portfolio
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Expected portfolio retuns per day
portfolio_returns = daily_returns.dot(weights)
#print(portfolio_returns)

# Answer 1

benchmark_exp_return = portfolio_returns
type(benchmark_exp_return)

benchmark_exp_return = float(portfolio_returns.mean())
benchmark_exp_return

########## the expected return as the average portfolio return (daily and yearly) ####################

#avg_portfolio_return = portfolio_returns.mean() # daily avg return
#print(avg_portfolio_return)

#benchmark_exp_return = avg_portfolio_return * 252 # assuming 252 trading days; yearly avg return
#print(portfolio_returns) # AVERAGE ANNUAL RETURN (Arithmetic mean)


##################################################################################################################################################################################################################################


# Answer 2

exp_return = portfolio_returns.mean() # exp return mean daily

daily_sharpe = exp_return / portfolio_returns.std()  # sharpe daily

annualized_sharpe = daily_sharpe * np.sqrt(252) # annualize it

benchmark_sharpe_ratio = annualized_sharpe
print(benchmark_sharpe_ratio)


############################### Annualized return ##############################

## Total compounded return per stock
## total_returns = (stock_prices_df.iloc[-1] - stock_prices_df.iloc[0]) / stock_prices_df.iloc[0]

## Total compounded for portfolio
#total_port_return = (portfolio_returns + 1).prod() - 1 # .prod() just sums all

## Now the ANNUALIZED return
#start_date = stock_prices_df.index[0]
#end_date = stock_prices_df.index[-1]
#years = (end_date - start_date).days / 365 # .days extracts the integer count of whole days in that interval

#annualized_return = ((1 + total_port_return)**(1/years))-1
##print(annualized_return)    # Annualized Return (Geometric Mean)

## Answer 2

############################## now benchmark_sharpe_ratio ##############################

# annualized volatility
#daily_volatility = portfolio_returns.std()    # calulate standard deviation

#ann_vol = daily_volatility * np.sqrt(252)
#risk_free = 0 # set risk free rate to 0

## the Sharpe ratio per stock
#benchmark_sharpe_ratio = annualized_return / ann_vol
#print(benchmark_sharpe_ratio)

##################################################################################################################################################################################################################################


# Answer 3

### Minimum Volatility portfolio

# First, calculate mu and Sigma
mu = expected_returns.mean_historical_return(stock_prices_df)
sigma = risk_models.sample_cov(stock_prices_df) # just a cov matrix of the stocks

# Then calculate min weights volatility
ef = EfficientFrontier(mu, sigma)
raw_weights = ef.min_volatility()
mv_portfolio = pd.Series(raw_weights)

# Extract volatility and store it as mv_portfolio_vol
ret, vol, sharpe = ef.portfolio_performance(verbose=False)
mv_portfolio_vol = vol


##################################################################################################################################################################################################################################

# Answer 4

### Maximum Sharpe portfolio

mu = expected_returns.mean_historical_return(stock_prices_df, compounding=False)
sigma = risk_models.sample_cov(stock_prices_df) # just a cov matrix of the stocks

# Ef is above, I just need (row) weights for Sharpe
ef = EfficientFrontier(mu, sigma)
raw_weights_sharpe = ef.max_sharpe(risk_free_rate=0)
ms_portfolio = pd.Series(raw_weights_sharpe)

# The max Sharpe ratio
ms_portfolio_sharpe = ef.portfolio_performance(risk_free_rate=0)[2]