import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import math

# Load the Churn Modelling dataset
df = pd.read_csv("Churn_Modelling.csv")
print(df.head())

# Preprocessing the dataset
# Encoding categorical variables: Gender and Geography
label_encoders = {}
for col in ['Gender', 'Geography']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Dropping irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Target variable: Exited
target = df['Exited']
features = df.drop(['Exited'], axis=1)

# Normalizing the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# Classification
X_train, X_test, y_train, y_test = train_test_split(df_scaled, target, train_size=0.8, random_state=42)

# K-Nearest Neighbors Classifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance evaluation for classification
print("\nClassification Metrics:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Error:", 1 - accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Specificity and False Positive Rate
specificity = tn / (tn + fp)
print("Specificity:", specificity)
print("False Positive Rate:", 1 - specificity)

# Plotting ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Regression for continuous variables
X = df_scaled['CreditScore'].values.reshape(-1, 1)
y = df_scaled['Balance']

# Karl Pearson's coefficient
corr, _ = pearsonr(X.flatten(), y)
print("\nRegression Metrics:")
print("Karl Pearson's Coefficient:", corr)

# Linear Regression
reg = LinearRegression()
reg.fit(X, y)
y_pred_reg = reg.predict(X)

# R-squared
r_squared = r2_score(y, y_pred_reg)
print("R-squared:", r_squared)

# Mean Squared Error
mse = mean_squared_error(y, y_pred_reg)
print("Mean Squared Error (MSE):", mse)

# Root Mean Squared Error
rmse = math.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Mean Absolute Error
mae = mean_absolute_error(y, y_pred_reg)
print("Mean Absolute Error (MAE):", mae)

# Mean Absolute Percentage Error
mape = (mae / max(y)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape, "%")
