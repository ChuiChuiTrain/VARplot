import pandas as pan
import numpy as nump
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf
from IPython import get_ipython
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller


maxlag = 12;

def grangers_causation_matrix(data, variables, test= 'ssr_chi2test', verbose = False):
    
    dataF = pan.DataFrame(nump.zeros((len(variables), len(variables))), columns = variables, index = variables)
    for column in dataF.columns:
        for row in dataF.index:
            test_result = grangercausalitytests(data[[row,column]], maxlag = maxlag, verbose = False);
            p_vals = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {row}, X = {column}, P Values = {p_vals}')
            min_p_val = nump.min(p_vals)
            dataF.loc[row, column] = min_p_val
    
    dataF.columns = [var + '_x' for var in variables]
    dataF.index = [var + '_y' for var in variables]
    return dataF
 
    
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    # TODO: generalize this to invert transformation for n differences
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:     
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


def forecast_accuracy(forecast, actual):
    
    mape = nump.mean(nump.abs(forecast - actual)/nump.abs(actual))  # MAPE
    me = nump.mean(forecast - actual)             # ME
    mae = nump.mean(nump.abs(forecast - actual))    # MAE
    mpe = nump.mean((forecast - actual)/actual)   # MPE
    rmse = nump.mean((forecast - actual)**2)**.5  # RMSE
    corr = nump.corrcoef(forecast, actual)[0,1]   # corr
    mins = nump.amin(nump.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = nump.amax(nump.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - nump.mean(mins/maxs)             # minmax
    
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})


def difference(dataFrame, sig = 0.05, count = 0):
    """perform ADF test on a dataframe until it is stationary
    returns a list which first item is the differenced dataframe
    and the second item is the number of times we differenced it"""
    isStationary = True
    
    for name, column in dataFrame.iteritems():
        r = adfuller(column, autolag = 'AIC')
        p_value = round(r[1],4)
        if p_value > sig:
            isStationary = False
    
    if(isStationary):
        return [dataFrame, count]
    else:
        df_differenced = dataFrame.diff().dropna()
        return difference(df_differenced, count = count + 1)

def adjust(val, length = 6):
        return str(val).ljust(length)

def cointegration_test(datafile, alpha=0.05):
    """P-Test with 0.05 cutoff using Johanson's Cointegration Test"""
    out = coint_johansen(datafile,-1,5)
    d = {'0.90':0,'0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)
    
    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20) 
    for col, trace, cvt in zip(datafile.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

        
def adfuller_test(series, signif = 0.05, name = "", verbose = False):
    """perform ADFuller to test stationarity of a series and print report"""
    r = adfuller(series, autolag = 'AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']

    
    #Print the summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    