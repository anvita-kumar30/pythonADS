import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.model_selection import GridSearchCV
from scipy.stats.mstats import winsorize

# Load dataset
file_path = 'supermarket_sales.csv'
df = pd.read_csv(file_path)

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract features from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear

# Encode categorical variables
label_encoder = LabelEncoder()
for col in ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']:
    df[col] = label_encoder.fit_transform(df[col])

# Drop non-relevant columns
df.drop(columns=['Invoice ID', 'Date', 'Time'], inplace=True, errors='ignore')

# Replace '-' with NaN and convert to numeric
df.replace('-', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop columns that have all missing values
df.dropna(axis=1, how='all', inplace=True)

# Identify numeric columns again after dropping missing ones
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Apply imputation only to valid numeric columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# 1. Explore Descriptive Statistics
print("Dataset Overview:\n", df.head())
print("\nSummary Statistics:\n", df.describe())

# 2. Data Visualization
plt.figure(figsize=(10, 5))
sns.histplot(df['Total'], bins=30, kde=True)
plt.title('Distribution of Total Sales')
plt.show()

# 3. Supervised Learning (Regression Model)
X = df.drop(columns=['Total'])
y = df['Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
model = grid_search.best_estimator_

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluation Metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)

# 4. Unsupervised Learning (Clustering)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='viridis')
plt.title("K-Means Clustering of Sales Data")
plt.show()

# 5. Time Series Forecasting
plt.figure(figsize=(12, 6))
plt.plot(df['Total'], label='Total Sales')
plt.title("Total Sales Over Time")
plt.xlabel("Index")
plt.ylabel("Total Sales")
plt.legend()
plt.show()

# 6. Outlier Detection (Winsorization)
df['Total'] = winsorize(df['Total'], limits=[0.05, 0.05])
print("Outliers handled using Winsorization.")

# 7. Inferential Statistics (Hypothesis Testing)
t_stat, p_value = stats.ttest_1samp(df['Total'], df['Total'].mean())
print("T-Test Statistic:", t_stat, " P-Value:", p_value)