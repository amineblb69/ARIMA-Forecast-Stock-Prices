# ARIMA-Forecast-Stock-Prices
This project implements a stock price forecasting pipeline using an ARIMA (AutoRegressive Integrated Moving Average) model.

This project provides a robust framework for time-series forecasting of stock prices using the ARIMA model. It features automated data retrieval, stationarity transformation, and model backtesting.

## Features
* **Automated Data Fetching**: Uses `yfinance` to pull historical market data.
* **Stationarity Engine**: Automatically applies log transformations and differencing to satisfy ARIMA requirements (Dickey-Fuller test validation).
* **Model Optimization**: Iterates through parameter combinations $(p, d, q)$ to find the best fit based on the **Akaike Information Criterion (AIC)**.
* **Rolling Backtest**: Evaluates model performance by simulating predictions on a historical test window.
* **Future Forecasting**: Projects future prices with 95% confidence intervals.

## Results Summary (Vistra Corp - VST)
The model was tested on VST data from 2020 to 2026.

### 1. Stationarity Analysis
The series required differencing to achieve stationarity, as evidenced by the ADF test and the stabilization of the moving average.

### 2. Autocorrelation (ACF & PACF)
These plots helped determine the initial lags for the AR and MA components.

### 3. Forecast vs. Actual
The model successfully tracked the trend during backtesting, with the final forecast indicating a continued growth trajectory.
