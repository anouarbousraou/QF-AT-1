# import packages
import pandas as pd
import statistics
from statistics import geometric_mean
import numpy as np
from scipy import stats
from numpy import NaN
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys

def load_data(data, fund_style=False):
    """ 
    Load data
    """
    df = pd.read_csv(data, index_col=0, parse_dates=True)
    strategies = df.iloc[:, 5:]
    if fund_style==True:
        return strategies
    else:
        return df.iloc[:,0:5]

# read data
hf_data = load_data("HedgeFund_Data.csv", fund_style=True)
ff_factors_data = load_data("HedgeFund_Data.csv", fund_style=False)

"""
Q2.1
"""
##A
ann_arith_mean = hf_data.mean()*12
print('***Annual arithmetic mean***', ann_arith_mean, sep='\n')

##B
# first do +1 for return, because geo mean cannot be calculated with negative values
hf_data_copy = hf_data+1
geo_means = []
for column in hf_data_copy:
    # calculate geo mean
    geo = stats.gmean(hf_data_copy[column])
    # annualize geo mean, source: https://math.stackexchange.com/questions/2191791/how-to-annualize-a-weekly-geometric-mean
    geo = geo**12 -1
    geo_means.append(geo)
ann_geo_mean = pd.Series(geo_means, index=list(hf_data.columns))
print('***Annual geometric mean***', ann_geo_mean, sep='\n')

##C
ann_volatility = hf_data.std()*12**.5
print('***Annual volatility***', ann_volatility, sep='\n')

##D

# create empty dataframe with index mdate
excess_returns = pd.DataFrame(index=ff_factors_data.index)
for column in hf_data:
    # compute excess returns per strategy
    excess_returns[column] = hf_data[column] - ff_factors_data['Rf']

ann_sharpe_ratio = excess_returns.mean()/excess_returns.std()*12**.5
print('***Annual Sharpe Ratio***', ann_sharpe_ratio, sep='\n')

##E
mkt_return = ff_factors_data['MktRf']
betas = []
alphas = []
for column in hf_data:
    # run regression for each hf strategy
    X = mkt_return
    Y = hf_data[column]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    res1 = model.fit(cov_type='HAC', cov_kwds={'maxlags':11})
    # get constant
    alpha = res1.params[0]
    alphas.append(alpha)
    # get beta
    beta = res1.params[1]
    betas.append(beta)

market_beta = pd.Series(betas, index=list(hf_data.columns))
print('***Market betas***', market_beta, sep='\n')

##F
ann_market_alpha = pd.Series(alphas, index=list(hf_data.columns))
ann_market_alpha = ann_market_alpha*12
print('***Annualized alpha to market***', ann_market_alpha, sep='\n')

##G
information_ratios = []

for col in excess_returns:
    # run regression based on excess returns
    Y = excess_returns[col].values
    # market returns were already defined earlier
    X = mkt_return
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    res2 = model.fit(cov_type='HAC', cov_kwds={'maxlags':11})
    alpha = res2.params[0]
    # get residuals, source: https://stackoverflow.com/questions/55095437/in-sklearn-regression-is-there-a-command-to-return-residuals-for-all-records
    residuals = res2.resid
    idio_sync_risk = residuals.std()
    # calculate information ratio
    information_ratio = alpha/idio_sync_risk
    information_ratios.append(information_ratio)

ann_info_ratio = pd.Series(information_ratios, index=list(hf_data.columns))  
print('***Annualized Information Ratio to market***', ann_info_ratio, sep='\n')

##H
# source: https://github.com/MISHRA19/Computing-Max-Drawdown-with-Python/blob/master/Max%20Drawdown.ipynb
t_draw_downs = []
for col in hf_data:
    return_index=(1+hf_data[col]).cumprod()
    previous_peaks=return_index.cummax()
    drawdown=(return_index-previous_peaks)/previous_peaks
    t_draw_downs.append(drawdown.min())
draw_downs = pd.Series(t_draw_downs, index=list(hf_data.columns))     
print('***Maximum drawdown***', draw_downs, sep='\n')

##I
skew = stats.skew(hf_data)
skewness = pd.Series(skew, index=list(hf_data.columns))
print('***Skewness***', skewness, sep='\n')

##J
# fisher subtracts 3 from kurtosis = excess??
# source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
kur = stats.kurtosis(hf_data, fisher=True)
kurtosis = pd.Series(kur, index=list(hf_data.columns))
print('***Excess Kurtosis***', kurtosis, sep='\n')

"""
Q2.2
"""

## A
cum_return = np.cumprod(1 + hf_data['Global_Macro']) - 1
# convert dates to datetime
cum_return.index = pd.to_datetime(cum_return.index, format='%Y%m', errors='coerce').dropna()

plt.plot(cum_return.index, cum_return.values)
plt.title('Cumulative return Global Macro Hedge Fund')
plt.ylabel('Cumulative returns in 100th %')
plt.xlabel('Time')
plt.show()

## B
return_index=(1+hf_data["Global_Macro"]).cumprod()
previous_peaks=return_index.cummax()
drawdown=(return_index-previous_peaks)/previous_peaks
drawdown.index = pd.to_datetime(drawdown.index, format='%Y%m', errors='coerce').dropna()
plt.plot(drawdown.index, drawdown.values)
plt.title('Drawdown Global Macro Hedge Fund')
plt.ylabel('Drawdown')
plt.xlabel('Time')
plt.show()

"""
Q2.3
"""

## 1st regression
X = mkt_return
Y = excess_returns["Long_Short"]
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results1 = model.fit(cov_type='HAC', cov_kwds={'maxlags':11})
print('***Regression question 2.3A***', results1.summary(), sep='\n')

## 2nd regression
factors = ff_factors_data

X = factors
X = sm.add_constant(X)
Y = excess_returns["Long_Short"]
model = sm.OLS(Y,X)
results2 = model.fit(cov_type='HAC', cov_kwds={'maxlags':11})
print('***Regression question 2.3B***', results2.summary(), sep='\n')

"""
Q2.4
"""
## A
X = mkt_return
# do we need to run it on excess returns or regular returns?
Y = hf_data['Convertible_Arb']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results3 = model.fit(cov_type='HAC', cov_kwds={'maxlags':11})
print('***Regression question 2.4A***', results3.summary(), sep='\n')

## B
conv_arb = hf_data['Convertible_Arb']
if len(conv_arb.values)%3 != 0:
     sys.exit('Program stops here, reason: Number of monthly return data is not in multiplication of 3')

# calculate quarterly returns by taking sum of monthly returns for 3 subsequent periods
quarterly_hf_returns = []
quarterly_mkt_returns = []  
# loops from 1 to 222
for i in range(1, int((len(conv_arb.values)+3)/3)):
    # picks starting value, this is always 3 periods before end period
    start = i*3-3
    # end period is in multiples of 3
    end = i*3
    # take sum
    quarterly_hf_return = sum(conv_arb.values[start:end])
    quarterly_hf_returns.append(quarterly_hf_return)

    quarterly_mkt_return = sum(mkt_return.values[start:end])
    quarterly_mkt_returns.append(quarterly_mkt_return)

Y = quarterly_hf_returns
X = quarterly_mkt_returns 
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results4 = model.fit(cov_type='HAC', cov_kwds={'maxlags':11})
print('***Regression question 2.4B***', results4.summary(), sep='\n')

## C
# create new data frame
market_returns = pd.DataFrame(mkt_return)
# create lagged returns
market_returns['One-lag'] = market_returns['MktRf'].shift(1)
market_returns['Two-lag'] = market_returns['MktRf'].shift(2)

# regression
beta_sum = 0 
# loop over different returns in market, from t to t-2 (lags)
for column in market_returns:
    model = sm.OLS(market_returns[column].values, hf_data['Convertible_Arb'].values, missing='drop')
    results5 = model.fit(cov_type='HAC', cov_kwds={'maxlags':11})
    beta = float(results5.params[0])
    # add beta to sum of beta
    beta_sum += beta
print("***Regression question 2.4C***")
print(f"The sum of betas of the Convertible Bond Arbitrage strategy is: {round(beta_sum, 3)}")
