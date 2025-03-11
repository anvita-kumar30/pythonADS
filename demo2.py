# CASE STUDY - Predicting Travel Time for Efficient Route Planning

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

# Data Cleaning (Handling Missing Values)
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Data Exploration
print("Dataset Overview:\n", df.head())
print("\nSummary Statistics:\n", df.describe())

# Data Visualization
plt.figure(figsize=(10, 5))
sns.histplot(df['TotalTime'], bins=30, kde=True)
plt.title('Distribution of Travel Time')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x=df['TotalTime'])
plt.title('Boxplot of Travel Time')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['Distance'], y=df['TotalTime'])
plt.title('Scatterplot of Distance vs Travel Time')
plt.show()

# Supervised Learning - KNN Classifier with Hyperparameter Tuning
X = df.drop(columns=['TotalTime'])
y = (df['TotalTime'] > df['TotalTime'].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Grid Search for best KNN parameters
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_knn = grid_search.best_estimator_

# Train KNN with best parameters
# best_knn.fit(X_train_scaled, y_train)
# y_pred = best_knn.predict(X_test_scaled)

# Classification Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC Curve
y_prob = knn.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Regression Model - RandomForest with Hyperparameter Tuning
X = df.drop(columns=['TotalTime'])
y = df['TotalTime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
model = grid_search.best_estimator_

# Train RandomForest with best parameters
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluation Metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance (in %):\n", (feature_importance * 100).round(2))

# Unsupervised Learning - Clustering
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='viridis')
plt.title("K-Means Clustering of Travel Data")
plt.show()

# Time Series Forecasting - Checking Stationarity
result = adfuller(df['TotalTime'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] > 0.05:
    print("Time series is non-stationary. Differencing is required.")

# Apply ARIMA Model
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

# Outlier Detection - DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
df['Outlier'] = dbscan.fit_predict(X)
outliers = df[df['Outlier'] == -1]
print("Detected Outliers:", outliers)

sns.scatterplot(x=df['Distance'], y=df['TotalTime'], hue=df['Outlier'], palette='coolwarm')
plt.title("DBSCAN Outlier Detection")
plt.show()

# Inferential Statistics - Hypothesis Testing
t_stat, p_value = stats.ttest_1samp(df['TotalTime'], df['TotalTime'].mean())
print("T-Test Statistic:", t_stat, " P-Value:", p_value)

from sklearn.neighbors import NearestNeighbors

# Standardizing the dataset for distance-based outlier detection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['TotalTime']))

# Fit Nearest Neighbors model
nearest_neighbors = NearestNeighbors(n_neighbors=5)
neighbors = nearest_neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# Sort distances for k-distance graph
sorted_distances = np.sort(distances[:, -1])

# Plot the K-Distance Graph
plt.figure(figsize=(10, 5))
plt.plot(sorted_distances)
plt.title("K-Distance Graph for Outlier Detection")
plt.xlabel("Data Points (sorted)")
plt.ylabel("Distance to 5th Nearest Neighbor")
plt.show()

# Detect Outliers - Choose Threshold (Elbow Point from Graph)
outlier_threshold = sorted_distances[int(len(sorted_distances) * 0.98)]  # Top 2% as outliers
outliers = np.where(distances[:, -1] > outlier_threshold)[0]
df['Outlier'] = 0
df.loc[outliers, 'Outlier'] = 1

# Visualizing Outliers
sns.scatterplot(x=df['Distance'], y=df['TotalTime'], hue=df['Outlier'], palette='coolwarm')
plt.title("K-Distance Outlier Detection")
plt.show()

print("Detected Outliers:", df[df['Outlier'] == 1])
