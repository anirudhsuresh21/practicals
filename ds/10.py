import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess AIS data
def load_and_preprocess(filepath):
    try:
        data = pd.read_csv(filepath)
        data['Time'] = pd.to_datetime(data['Time'])
        data = data.sort_values('Time')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train-test split
def split_data(ts_data, train_ratio=0.8):
    train_size = int(len(ts_data) * train_ratio)
    train, test = ts_data[:train_size], ts_data[train_size:]
    return train.fillna(method='ffill'), test.fillna(method='ffill')

# Fit and forecast using ARIMA
def arima_forecast(train, test, order=(2,1,1)):
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(len(test))
        return forecast
    except Exception as e:
        print(f"ARIMA model error: {e}")
        return None

# Fit and forecast using SARIMAX
def sarimax_forecast(train, test, order=(2,1,1), seasonal_order=(1,1,1,6)):
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(len(test))
        return forecast
    except Exception as e:
        print(f"SARIMAX model error: {e}")
        return None

# Plot function
def plot_results(train, test, arima_forecast, sarimax_forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-50:], train[-50:], label='Train')
    plt.plot(test.index, test, label='Test')
    if arima_forecast is not None:
        plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
    if sarimax_forecast is not None:
        plt.plot(test.index, sarimax_forecast, label='SARIMAX Forecast')
    plt.legend()
    plt.title("AIS Speed Over Ground Forecasting")
    plt.xlabel("Time")
    plt.ylabel("Speed (knots)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Error metrics
def calculate_errors(test, forecast, model_name):
    if forecast is not None:
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"{model_name} MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Main execution
filepath = "SolentAIS_20160112_1302111.csv"
data = load_and_preprocess(filepath)
if data is not None:
    ts_data = data.groupby('Time')['COG_degrees'].mean()
    train, test = split_data(ts_data)
    
    arima_pred = arima_forecast(train, test)
    sarimax_pred = sarimax_forecast(train, test)
    
    plot_results(train, test, arima_pred, sarimax_pred)
    
    calculate_errors(test, arima_pred, "ARIMA")
    calculate_errors(test, sarimax_pred, "SARIMAX")