import pandas as pan
import numpy as nump
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from IPython import get_ipython
from helper_functions import grangers_causation_matrix, invert_transformation, forecast_accuracy, difference, adfuller_test, cointegration_test, adjust
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson

register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')

# get filepath via input
filepath = input("Enter file path: ");

# load data using pandas
dataFrame = pan.read_excel(filepath, parse_dates= ['Years'], index_col= "Years");
print(dataFrame.shape);
dataFrame.tail();

# plot the data
fig, axes = plt.subplots(nrows=7, ncols=1, dpi=120, figsize=(10,15))

for i, ax in enumerate(axes.flatten()):
    
    data = dataFrame[dataFrame.columns[i]]
    
    ax.plot(data, color='red', linewidth=1)
    
    # Decorations
    ax.set_title(dataFrame.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();

nobs = 4
df_train, df_test = dataFrame[0:-nobs], dataFrame[-nobs:]

          
# print the sizes
print(df_train.shape)
print(df_test.shape)

# perform ADF test on each column of the 1st diff dataframe
df_differenced_val = difference(df_train)
df_differenced = df_differenced_val[0]
df_differenced_count = df_differenced_val[1]

# perform ADF test on each column of 2nd diff dataframe
#df_differenced = df_differenced.diff().dropna()

#perform ADF test on each column of 3rd diff dataframe
#df_differenced = df_differenced.diff().dropna()

#perform ADF test on 4th diff dataframe to get it to be stationary

#df_differenced = df_differenced.diff().dropna()

for name, column in df_differenced.iteritems():
    adfuller_test(column, name = column.name)
    print('\n')
    

print(grangers_causation_matrix(dataFrame, variables = dataFrame.columns))
cointegration_test(dataFrame)

# select the order of VAR model
model = VAR(df_differenced)

for i in range(1,10):
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

x = model.select_order(maxlags = 10)
print(x.summary())

model_fitted = model.fit(10)
print(model_fitted.summary())

# use Durbin Watson Statistic to check for errors
out = durbin_watson(model_fitted.resid)

for col, val in zip(dataFrame.columns, out):
    print(adjust(col), ':', round(val,2))

# get the lag order
lag_order = model_fitted.k_ar
print(lag_order)

# input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
print(forecast_input)

# Forecast
fc = model_fitted.forecast(forecast_input, nobs)
df_forecast = pan.DataFrame(fc, index = dataFrame.index[-nobs:], columns = dataFrame.columns + '_2d')

print(df_forecast)

# invert transformations to get real forecast
df_results = invert_transformation(df_train, df_forecast, second_diff = True)

temp_list = []
for item in dataFrame.columns:
    temp_list.append(item + '_forecast')
df_results.loc[:, temp_list]

print(df_results)

fig, axes = plt.subplots(nrows=int(len(dataFrame.columns)), ncols=1, dpi=150, figsize=(10,15))
for i, (col,ax) in enumerate(zip(dataFrame.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();

print('Forecast Accuracy of: Annual Growth')
accuracy_prod = forecast_accuracy(df_results['Annual Growth _forecast'].values, df_test['Annual Growth '])
for k, v in accuracy_prod.items():
    print(adjust(k),': ',round(v,4))
    
