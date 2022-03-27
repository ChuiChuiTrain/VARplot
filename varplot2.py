import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import calendar
from datetime import datetime
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf

# read in the csv file
raw_t = pd.read_csv('./data/GLB.Ts+dSST.csv', skiprows = 1)

# create new dataframe with an index for each month
date_rng = pd.date_range(start = '1/1/1880', end = '1/03/2019', freq='M')
t = pd.DataFrame(date_rng, columns = ['date'])
t['Avg_Anomaly_deg_C'] = None
t.set_index('date', inplace = True)
# remove the columns after Dec because we are not interested in them
raw_t = raw_t.iloc[:,:13]

# populate the dataframe with the entries of raw_t
def populate_df_with_anomolies_from_row(row):
    year = row['Year']
    monthly_anomolies = row.iloc[1:]

    for month in monthly_anomolies.index:
        last_day = calendar.monthrange(year, datetime.strptime(month, '%b').month)[1]
        date_index = datetime.strptime(f'{year} {month} {last_day}','%Y %b %d')
        t.loc[date_index] = monthly_anomolies[month]

raw_t.apply(lambda row: populate_df_with_anomolies_from_row(row), axis = 1)


# cleanup NaN values
def clean_up(raw_value):
    try:
        return float(raw_value)
    except:
        return np.NaN
    
t['Avg_Anomaly_deg_C'] = t['Avg_Anomaly_deg_C'].apply(lambda raw_value: clean_up(raw_value))

# forward fill to take care of NaN
t.fillna(method = 'ffill', inplace = True)

# Plot the time series
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.xlabel('Time')
plt.ylabel('Temperature Anomaly (C)')
plt.plot(t, color='#1C7C54', linewidth=1.0)

# From this plot this is clearly not stationary, the mean is changing
# Just in case find the ACF
# Note it is possible to do an ADF test here instead to determine if it is stationary

plot_acf(t)

# WOW WHAT A SURPRISE! ITS NOT STATIONARY (sarcasm intended)
# the slow decay in the ACF indicates that differencing may be needed
# Try to detrend the time series by differencing
# we might try to detrend using regression, but this does not look quite linear
# so it is unlikely that this will lead to a stationary TS

t_diff = t.diff()

plt.figure(figsize=(10,8))
plt.xlabel('Time')
plt.ylabel('Temperature Anomaly (C) (differenced)')
plt.plot(t, color='#1C7C54', linewidth=1.0)

# check if stationary
t_diff = t_diff.dropna()
print(t_diff.iloc[:,0].values)
result = adfuller(t_diff.iloc[:,0].values, autolag = 'AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# our ADF statistic is -12.2 which means that this is likely stationary
# Furthermore, the p-value is 0.00000 which means it is likely stationary

plot_acf(t_diff)
plot_pacf(t_diff)

                    
# since it is a univariate time series ie we only have one time series
# Also note that there does not appear to be seasonal patterns
# and we are not predicting using more we should use either ARIMA or ARMA model
# since we already differenced an ARMA model seems to be most appropriate
# From the ACF and PACF I believe that ARMA(1,1) should be sufficient
# note that if we do have multiple time series we can use VARIMA
# maybe we can use CO2 emmisions to predict future in conjunction with current time series
# Note that this is technically an ARIMA(1,1,1) model since we differenced once
t = t.dropna()
model = ARIMA(t.iloc[:,0].values, order=(2,1,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# Plotting residuals to check for constant mean and variance
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Create training and test
train = t.iloc[:,0][:1650]
test = t.iloc[:,0][1650:]

# Build model
# with order I=1 our forecast does not satisfy.
# It is consistently lower than the actual and some values are out of the
# confidence band.
model = ARIMA(train, order=(1,1,1))
# model = ARIMA(train, order=(1,2,0))
fitted = model.fit(disp=-1)

fc, se, conf = fitted.forecast(30, alpha=.05)

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train.iloc[1634:], label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# forecast accuracies 
# Typically, if you are comparing forecasts of two different series, the MAPE, Correlation and Min-Max Error can be used
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

print(forecast_accuracy(fc, test.values))
