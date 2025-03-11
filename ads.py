import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
file_path = 'travel-times.csv'
df = pd.read_csv(file_path)

# Convert Date and StartTime to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M', errors='coerce').dt.hour

# Extract features from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear

# Encode categorical variables
label_encoder = LabelEncoder()
df['DayOfWeek'] = label_encoder.fit_transform(df['DayOfWeek'])
df['Take407All'] = label_encoder.fit_transform(df['Take407All'])
df['GoingTo'] = label_encoder.fit_transform(df['GoingTo'])

# Drop non-relevant columns
df.drop(columns=['Comments', 'Date'], inplace=True)

# Replace '-' with NaN and convert to numeric
df.replace('-', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values with median
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Feature Scaling
scaler = StandardScaler()
X = df.drop(columns=['TotalTime'])
X_scaled = scaler.fit_transform(X)

# Finding the optimal number of clusters using Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choosing the optimal k (adjust based on elbow method)
optimal_k = 4  # Change this based on the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing Clusters
sns.scatterplot(x=df['Distance'], y=df['TotalTime'], hue=df['Cluster'], palette='viridis')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.show()

# Evaluating K-Means using Silhouette Score
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f'Silhouette Score: {sil_score:.3f}')

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Tune eps and min_samples
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Visualizing DBSCAN Clusters
sns.scatterplot(x=df['Distance'], y=df['TotalTime'], hue=df['DBSCAN_Cluster'], palette='coolwarm')
plt.title('DBSCAN Clustering')
plt.show()

# Checking Stationarity for Time Series Forecasting
result = adfuller(df['TotalTime'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] > 0.05:
    print("Time series is non-stationary. Differencing is required.")

# ARIMA Model
plt.figure(figsize=(12, 6))
plt.plot(df['TotalTime'], label='Travel Time')
plt.title("Travel Time Over Time")
plt.xlabel("Index")
plt.ylabel("Travel Time")
plt.legend()
plt.show()

model = ARIMA(df['TotalTime'], order=(2,1,2))
model_fit = model.fit()
predictions = model_fit.predict(start=len(df)-50, end=len(df)-1, typ='levels')

plt.figure(figsize=(12, 6))
plt.plot(df['TotalTime'], label='Actual')
plt.plot(predictions, label='Predicted', color='red')
plt.title("ARIMA (2,1,2) Model Forecast")
plt.legend()
plt.show()

# Outlier Detection using DBSCAN
dbscan_outlier = DBSCAN(eps=3, min_samples=5)
df['Outlier'] = dbscan_outlier.fit_predict(X_scaled)
outliers = df[df['Outlier'] == -1]
print("Detected Outliers:", outliers)

sns.scatterplot(x=df['Distance'], y=df['TotalTime'], hue=df['Outlier'], palette='coolwarm')
plt.title("DBSCAN Outlier Detection")
plt.show()

# Hypothesis Testing
t_stat, p_value = stats.ttest_1samp(df['TotalTime'], df['TotalTime'].mean())
print("T-Test Statistic:", t_stat, " P-Value:", p_value)