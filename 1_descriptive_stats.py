import pandas as pd

# Load the dataset
file_path = "diamonds.csv"
df = pd.read_csv(file_path)

# Display basic information and the first few rows
df.info(), df.head()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

# Select relevant numerical columns
num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
df_num = df[num_cols]

# Descriptive statistics
desc_stats = df_num.describe()

# Additional statistics
mode_values = df_num.mode().iloc[0]  # First row of mode values
median_values = df_num.median()
variance = df_num.var()
skewness = df_num.skew()
kurtosis = df_num.kurtosis()
sem_values = df_num.sem()
missing_values = df_num.isnull().sum()
trimmed_mean_values = {col: trim_mean(df_num[col], 0.1) for col in num_cols}

# Scatter plot example (carat vs price)
plt.scatter(df['carat'], df['price'], c='blue', alpha=0.5)
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Scatter Plot of Carat vs Price')
plt.show()

# Boxplot for price distribution
plt.figure(figsize=(10, 5))
plt.boxplot(df['price'], vert=False)
plt.xlabel('Price')
plt.title('Boxplot of Diamond Prices')
plt.show()

# Display results
desc_stats, mode_values, median_values, variance, skewness, kurtosis, sem_values, missing_values, trimmed_mean_values
