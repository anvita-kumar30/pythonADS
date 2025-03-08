# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("travel-times.csv")

# Convert 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sort the data by date
df = df.sort_index()

# Selecting 'TotalTime' as the target variable for forecasting
ts = df['TotalTime']

# Plot the time series
plt.figure(figsize=(12, 5))
plt.plot(ts, label="Total Travel Time")
plt.title("Travel Time Over Time")
plt.xlabel("Date")
plt.ylabel("Total Time (minutes)")
plt.legend()
plt.show()

# Decomposing the time series
decomposition = seasonal_decompose(ts, model='additive', period=30)
decomposition.plot()
plt.show()

# Augmented Dickey-Fuller Test for stationarity
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("The data is non-stationary.")
    else:
        print("The data is stationary.")

adf_test(ts)  # The test already shows it's stationary

# Fit ARIMA model (No differencing needed, so d=0)
model = ARIMA(ts, order=(0,1,1))  # Adjusted based on previous results
arima_result = model.fit()
print(arima_result.summary())

# Train-Test Split (80% Training, 20% Testing)
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Fit the model on training data
train_model = ARIMA(train, order=(0,1,1)).fit()

# Forecast the test period
test_forecast = train_model.forecast(steps=len(test))

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Test Data", color='blue')
plt.plot(test.index, test_forecast, label="Forecasted Test Data", color='red')
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()

# Model Evaluation on test set
mae = mean_absolute_error(test, test_forecast)
mse = mean_squared_error(test, test_forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test - test_forecast) / test)) * 100

print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Future Forecasting
forecast_steps = 10
future_forecast = arima_result.forecast(steps=forecast_steps)

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(ts, label="Actual")
plt.plot(pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='D')[1:], future_forecast, label="Future Forecast", color='green')
plt.legend()
plt.title("Future ARIMA Forecast for Travel Time")
plt.show()
