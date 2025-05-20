### Preparatory work


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns, EfficientFrontier, plotting


# Setting the plotting style to be colorblind-friendly
plt.style.use("seaborn-v0_8-colorblind")

# Loading data
stock_prices_df = pd.read_csv("faang_stocks.csv", index_col="Date")

# Changing the index to a datetime type allows for easier filtering and plotting.
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)
stock_prices_df


####################################################################################################################################################################################
# 1. Capture the Axes returned by DataFrame.plot()
ax = stock_prices_df.plot(
    title="FAANG stock prices from years 2020–2023",
    ylabel="Price (USD)"
)

# 2. Turn off the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (Optional) tighten up the layout
plt.tight_layout()
plt.show()






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

# Total compounded return per stock
# total_returns = (stock_prices_df.iloc[-1] - stock_prices_df.iloc[0]) / stock_prices_df.iloc[0]

# Total compounded for portfolio
total_port_return = (portfolio_returns + 1).prod() - 1 # .prod() just sums all

# Now the ANNUALIZED return
start_date = stock_prices_df.index[0]
end_date = stock_prices_df.index[-1]
years = (end_date - start_date).days / 365 # .days extracts the integer count of whole days in that interval

annualized_return = ((1 + total_port_return)**(1/years))-1
annualized_return    # Annualized Return (Geometric Mean)

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
mv_portfolio_vol


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

####################################################################
# plot frontier


# 1. Recompute inputs
mu    = expected_returns.mean_historical_return(stock_prices_df)
sigma = risk_models.sample_cov(stock_prices_df)

# 2. Instantiate frontier object
ef = EfficientFrontier(mu, sigma)

# 3. Compute the two special portfolios
ef_min = EfficientFrontier(mu, sigma)
ef_min.min_volatility()
ret_min, vol_min, _ = ef_min.portfolio_performance(verbose=False, risk_free_rate=0)

ef_shp = EfficientFrontier(mu, sigma)
ef_shp.max_sharpe(risk_free_rate=0)
ret_shp, vol_shp, _ = ef_shp.portfolio_performance(verbose=False, risk_free_rate=0)

# 4. Plot everything
fig, ax = plt.subplots(figsize=(8, 6))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# 4a. Plot and label each individual stock
for ticker in stock_prices_df.columns:
    vol_i = np.sqrt(sigma.loc[ticker, ticker])
    ret_i = mu[ticker]
    ax.scatter(vol_i, ret_i, color="k", s=30)
    ax.annotate(
        ticker,
        (vol_i, ret_i),
        textcoords="offset points",
        xytext=(5, 5),
        ha="left",
        fontsize=9
    )

# 4b. Highlight min-vol and max-Sharpe portfolios
ax.scatter([vol_min], [ret_min], marker="*", s=200, label="Min Vol", edgecolor="k")
ax.scatter([vol_shp], [ret_shp], marker="*", s=200, label="Max Sharpe", edgecolor="k")

# 4c. Draw the Capital Market Line (CML) from (0,0) → (vol_shp, ret_shp)
ax.plot([0, vol_shp], [0, ret_shp], linestyle="--", linewidth=1, label="CML (rf=0%)")

# 4d. Beautify
ax.grid(True, alpha=0.3)
ax.set_title("Efficient Frontier – FAANG Portfolio")
ax.set_xlabel("Annualized Volatility (Risk)")
ax.set_ylabel("Annualized Expected Return")
ax.legend()

plt.tight_layout()
plt.show()