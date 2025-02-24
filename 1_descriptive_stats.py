import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import trim_mean, skew, kurtosis, variation, sem, zscore, norm, poisson, ttest_ind, ttest_1samp
from scipy.stats import f_oneway

# Load the dataset
file_path = "diamonds.csv"
df = pd.read_csv(file_path)

# Display basic information and the first few rows
df.info()
print(df.head())

# Select relevant numerical columns
num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
df_num = df[num_cols]

# Descriptive statistics
desc_stats = df_num.describe()
mode_values = df_num.mode().iloc[0]  # First row of mode values
median_values = df_num.median()
variance = df_num.var()
skewness = df_num.skew()
kurtosis_values = df_num.kurtosis()
sem_values = df_num.sem()
missing_values = df_num.isnull().sum()
trimmed_mean_values = {col: trim_mean(df_num[col], 0.1) for col in num_cols}

# Additional descriptive statistics
range_values = df_num.max() - df_num.min()
q1 = df_num.quantile(0.25)
q3 = df_num.quantile(0.75)
iqr = q3 - q1
correlation_matrix = df_num.corr()
cv = variation(df_num, axis=0)
n_total = len(df)
cumulative_n = np.cumsum(df_num.count())
percent = (df_num.count() / n_total) * 100
cumulative_percent = np.cumsum(percent)
sum_of_squares = np.sum(df_num ** 2, axis=0)

# Visualizations
plt.figure(figsize=(10, 5))
plt.scatter(df['carat'], df['price'], c='blue', alpha=0.5)
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Scatter Plot of Carat vs Price')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['price'])
plt.title('Boxplot of Diamond Prices')
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Inferential Statistics
# Normal and Poisson Distributions
mu, sigma = df['price'].mean(), df['price'].std()
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, label='Normal Distribution')
plt.title('Normal Distribution of Diamond Prices')
plt.legend()
plt.show()

lambda_poisson = df['price'].mean()
x_poisson = np.arange(0, lambda_poisson * 2)
y_poisson = poisson.pmf(x_poisson, lambda_poisson)
plt.bar(x_poisson, y_poisson, alpha=0.6, color='b', label='Poisson Distribution')
plt.title('Poisson Distribution of Diamond Prices')
plt.legend()
plt.show()

# Hypothesis Testing
# Z-Test (Assuming a large sample size)
z_scores = zscore(df['price'])

# T-Test: Compare mean price of diamonds above and below 1 carat
above_1carat = df[df['carat'] > 1]['price']
below_1carat = df[df['carat'] <= 1]['price']
t_stat, p_val = ttest_ind(above_1carat, below_1carat)

# One-sample t-test
pop_mean = df['price'].mean()
t_stat_1samp, p_val_1samp = ttest_1samp(df['price'], pop_mean)

# ANOVA Test
anova_stat, anova_p_val = f_oneway(df['price'], df['carat'])

# Display results
print("Descriptive Statistics:", desc_stats)
print("Mode:", mode_values)
print("Median:", median_values)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis:", kurtosis_values)
print("Standard Error of Mean:", sem_values)
print("Missing Values:", missing_values)
print("Trimmed Mean:", trimmed_mean_values)
print("Range:", range_values)
print("Interquartile Range:", iqr)
print("Correlation Matrix:\n", correlation_matrix)
print("Coefficient of Variation:", cv)
print("N Total:", n_total)
print("Cumulative N:", cumulative_n)
print("Percent:", percent)
print("Cumulative Percent:", cumulative_percent)
print("Sum of Squares:", sum_of_squares)

print("\nInferential Statistics:")
print("T-Test Between Carat Groups: t-stat:", t_stat, ", p-value:", p_val)
print("One-Sample T-Test: t-stat:", t_stat_1samp, ", p-value:", p_val_1samp)
print("ANOVA Test: F-stat:", anova_stat, ", p-value:", anova_p_val)
