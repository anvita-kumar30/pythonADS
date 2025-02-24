import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("diamonds.csv")

# Drop unnecessary column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['cut', 'color', 'clarity']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['price'])  # Clustering features
y = df['price'] >= df['price'].median()  # Binarize price based on median

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
y_pred = kmeans.fit_predict(X_scaled)

# Evaluate Clustering using the mentioned metrics
print("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))  # Extrinsic Measure
print("Mutual Information Score:", normalized_mutual_info_score(y, y_pred))  # Extrinsic Measure
print("Silhouette Score:", silhouette_score(X_scaled, y_pred))  # Intrinsic Measure
