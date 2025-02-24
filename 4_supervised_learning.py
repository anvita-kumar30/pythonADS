import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import math
import numpy as np

# Load the Churn Modelling dataset
df = pd.read_csv("Churn_Modelling.csv")

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
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)  # Sensitivity
specificity = tn / (tn + fp)
fpr = 1 - specificity  # False positive rate
f1 = f1_score(y_test, y_pred)
geometric_mean = math.sqrt(recall * specificity)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Error Rate: {error_rate}")
print(f"Precision: {precision}")
print(f"Recall (Sensitivity): {recall}")
print(f"Specificity: {specificity}")
print(f"False Positive Rate: {fpr}")
print(f"F1 Score: {f1}")
print(f"Geometric Mean: {geometric_mean}")
print(f"ROC AUC Score: {roc_auc}")

# Plotting ROC Curve
fpr_values, tpr_values, _ = roc_curve(y_test, y_pred)
plt.plot(fpr_values, tpr_values, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Regression for continuous variables
X = df_scaled['CreditScore'].values.reshape(-1, 1)
y = df_scaled['Balance']

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
mape = np.mean(np.abs((y - y_pred_reg) / y)) * 100  # Mean Absolute Percentage Error
rmsre = np.sqrt(np.mean((y - y_pred_reg) ** 2) / np.mean(y ** 2))  # Root Mean Squared Relative Error

print(f"R-squared: {r_squared}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Root Mean Squared Relative Error (RMSRE): {rmsre}")
