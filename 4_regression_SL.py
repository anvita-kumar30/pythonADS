import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv("swedish_insurance.csv")

# X= number of insurance claims
# Y= amount paid in thousands for the claims

# Define features and target variable
X = df['X'].values.reshape(-1, 1)  # Number of claims
y = df['Y'].values  # Amount paid

# Pearson Correlation Coefficient
corr, _ = pearsonr(X.flatten(), y)
print("\nRegression Metrics:")
print("Pearson Correlation Coefficient:", corr)

# Linear Regression
reg = LinearRegression()
reg.fit(X, y)
y_pred_reg = reg.predict(X)

# Regression performance metrics
r_squared = r2_score(y, y_pred_reg)
mse = mean_squared_error(y, y_pred_reg)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y, y_pred_reg)
mape = np.mean(np.abs((y - y_pred_reg) / np.where(y == 0, 1e-10, y))) * 100  # Handling zero values in y
rmsre = np.sqrt(np.mean((y - y_pred_reg) ** 2) / np.mean(y ** 2))  # Root Mean Squared Relative Error

# Print metrics
print(f"R-squared: {r_squared}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Root Mean Squared Relative Error (RMSRE): {rmsre}")

# Scatter plot with regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred_reg, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Number of Claims")
plt.ylabel("Amount Paid (in thousands)")
plt.title("Insurance Claims vs Amount Paid")
plt.legend()
plt.show()