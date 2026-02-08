import yfinance as yf
import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()

ticker='VST'
start='2020-01-01'
end='2026-01-26'
interv='1wk'

def get_data(tick: str = None, startdate: str = None, enddate: str = None, interv=None, adjusted: bool = True) :

    """Collect stock values from Yahoo Finance and compute daily returns.
    """
    if tick is None:
        tick = input("Enter the ticker (e.g. AAPL): ").strip()
    if startdate is None:
        startdate = input("Enter the start date (YYYY-MM-DD): ").strip()
    if enddate is None:
        enddate = input("Enter the end date (YYYY-MM-DD): ").strip()
    if interv is None:
        interv='1d'

    df = pd.DataFrame(yf.download(
        tickers=tick,
        start=startdate,
        end=enddate,
        interval=interv,
        progress=False
        ))

    if df.empty:
        raise ValueError(f"No data retrieved for ticker '{tick}' between {startdate} and {enddate}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price_col = 'Adj Close' if (adjusted and 'Adj Close' in df.columns) else 'Close'


    df = df[[price_col]].rename(columns={price_col: 'Close'}).copy()


    df['Returns'] = df['Close'].pct_change()
    df['Log_Close'] = np.log(df['Close'])

    df.columns.name=None

    return df


df = get_data(ticker,start,end,interv=interv)
print(df)

def test_stationarity(timeserie):

    """Realise a Dickey-Fuller test to define if the timeserie is stationary or not."""

    ts = timeserie.dropna()
    result = adfuller(ts)
    print('Stats ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeures Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    return result

def get_stationary(timeserie):
    """Get the timeserie stationary if not."""
    ts = timeserie.dropna()
    result = adfuller(ts)
    if result[1] < 0.05:
        return ts, 0  # Stationary already

    # Log transform check
    if (ts > 0).all().item():
        ts_log = np.log(ts)
        result = adfuller(ts_log)
        if result[1] <= 0.05:
            print("The serie is stationary after log")
            return ts_log, 0
        ts = ts_log
    else:
        print("The serie include negative values, no log")

    ts_diff = ts.copy()
    
    for d in range(1, 4):
        ts_diff = ts_diff.diff().dropna()
        result = adfuller(ts_diff)
        if result[1] <= 0.05:
            print(f"The serie is stationary after {d} diffs")
            return ts_diff, d 
    
    
    print("Aucune transformation n'a rendu la série stationnaire. Retour de d=1 par défaut.")
    return ts.diff().dropna(), 1 

df_stationary , d_value = get_stationary(df['Close'])


rolling_mean1=df_stationary.rolling(window=100).mean()
rolling_std1=df_stationary.rolling(window=100).std()


plt.plot(df_stationary, color='blue', label='stationary serie')
plt.plot(rolling_mean1, color='red', label='MA')
plt.plot(rolling_std1, color='black', label='ecart type')
plt.legend()
plt.show()


#____STATIONARY_TEST_RESULTS___
print("\n=== Stationarity Test Results ===")
print(test_stationarity(df_stationary))

''''''
#____PLOT PACF AND ACF___

fig, axes = plt.subplots(1,2,figsize=(14,4))
plot_acf(df_stationary.dropna(), lags=40, ax=axes[0])
plot_pacf(df_stationary.dropna(), lags=40,ax=axes[1])
axes[0].set_title('Autocorelation Function (ACF)')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.show()

#___ARIMA MODEL APPLY__

#Set data to train & test
train_size= int(len(df)*0.8)
train, test= df[:train_size]['Log_Close'] , df[train_size:]['Log_Close']


def select_model(train,d):
    """Select the best ARIMA model baised on AIC"""

    p_serie= range(0,3)
    q_serie= range(0,3)

    best_aic=np.inf
    best_parameters=(0,d_value,0) #Default ARIMA parameters

    trend_options1=['t','n']
    trend_options2=['c','ct','ctt']

    if d==0:
      trend_options=trend_options2
    else:
      trend_options=trend_options1


    for p , q, t in product(p_serie,q_serie,trend_options):
        if p==0 and q==0 :
            continue
        try :
            model = ARIMA(train, order=(p,d,q),trend=t)
            results = model.fit()
            if results.aic < best_aic :
                best_aic = results.aic
                best_parameters = (p,d,q)
                best_tvalue = t
        except Exception as e :
            print(f"Erreur pour ARIMA({p},{d_value},{q}): {str(e)[:50]}")
            continue

    print(f"The best order for ARIMA parameter is : {best_parameters} with AIC : {best_aic:.2f}")
    print(f"\nThe best trend parameter is : {best_tvalue}")
    return best_parameters, best_tvalue, best_aic

best_order,t_value,aic_results=select_model(train,d_value)

def backtest_arima(df,best_order,window_size=None,step=1,trend=None):
    """Perform rolling window backtesting for ARIMA model."""

    predictions=[]
    actuals=[]
    dates=[]
    gap=[]

    # Accept either a Series of prices or a DataFrame with a 'Close' column
    if isinstance(df, pd.Series):
        prices = df
    else:
        if 'Log_Close' not in df.columns:
            raise ValueError("df must contain 'Log_Close' column or be a Series of prices")
        prices = df['Log_Close']

    n = len(prices)
    local_train_size = int(n * 0.8)

    print(f"\n=== Launching Backtest ===")
    print(f"\n=== Train dataset size : {local_train_size} observations ===")
    print(f"\n=== Test dataset size : {n - local_train_size} observations ===")

    for i in range(local_train_size, n - step + 1):

        if window_size is None:
            # expanding window: use all available observations up to i
            train_data = prices[:i]
        else:
            # rolling window: ensure positive integer
            try:
                ws = int(window_size)
                if ws <= 0:
                    raise ValueError
            except Exception:
                raise ValueError("window_size must be a positive integer or None")
            train_data = prices[max(0, i - ws):i]

        try :
            model = ARIMA(train_data, order=best_order, trend=trend)
            fitted_model = model.fit()

            forecast = fitted_model.forecast(steps=step)
            pred_value = forecast.iloc[-1] if isinstance(forecast, pd.Series) else forecast[-1]

            predictions.append(pred_value)
            actuals.append(prices.iloc[i + step - 1])
            dates.append(prices.index[i + step - 1])
            gap.append((predictions[-1]-actuals[-1])/actuals[-1])
            if (i - local_train_size) % 50 == 0:
                print(f"Progress: {i - local_train_size}/{n - local_train_size} forecasts completed")
        except Exception as e:
            print(f"Error at index {i}: {str(e)[:200]}")
            continue

    results=pd.DataFrame({
        'Date': dates,
        'Actual Log': actuals,
        'Predicted Log': predictions,
        'Actual':np.exp(actuals),
        'Predicted':np.exp(predictions),
        'Gap':gap
    })

    results=results.set_index('Date')
    return results

results_arima_backtest = backtest_arima(df, best_order,trend=t_value)

df['Predicted']=results_arima_backtest['Predicted']
df['Gap']=results_arima_backtest['Gap']
gap_mean=df['Gap'].mean()

def run_arima(df, best_order, n_steps=None,trend=None):
    forecasts = []
    forecast_dates = []

    if n_steps is None:
        n_steps = int(len(df)*0.2)

    working_data = df['Log_Close'].copy()

    model=ARIMA(working_data,order=best_order,trend=trend)
    fitted_model=model.fit()
    forecasts_obj=fitted_model.get_forecast(n_steps)

    forecasts = forecasts_obj.predicted_mean
    conf_int=forecasts_obj.conf_int(alpha=0.05)

    last_date=df.index[-1]
    forecast_dates=pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_steps, freq='B')




    results = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast_Log': forecasts,
        'Forecast_Price': np.exp(forecasts),
        'Lower Price':np.exp(conf_int.iloc[:, 0].values),
        'Upper Price':np.exp(conf_int.iloc[:, 1].values)
    })

    results=results.set_index('Date',inplace=False)

    return results

results_run=run_arima(df,best_order,trend=t_value)
df_final=pd.concat([df,results_run])
print(df_final)

print(f"The best order for ARIMA parameter is : {best_order} with AIC : {aic_results:.2f}")
print(f"\nThe best trend parameter is : {t_value}")


#PLOT ARIMA MODEL RESULTS 
fig, (ax1b,ax2b) = plt.subplots(1,2,figsize=(16,6))

ax1b.plot(df['Close'],label='Acutal Close',color='black')
ax1b.plot(df['Predicted'],'--',label='Backtest',color='orange',)
ax1b.plot(df_final['Forecast_Price'],label='Forecast',color='red')
ax1b.fill_between(df_final.index, df_final['Lower Price'], df_final['Upper Price'], color='grey', alpha=0.3)
ax1b.set_title(f'Result ARIMA model on {ticker}')
ax1b.legend()

ax2b.hist(df['Gap'],color='darkblue',label=f'Mean = {gap_mean:.2f}')
ax2b.set_title('Distrib of Gap Between Acutal an Predicted Price')

plt.show()
