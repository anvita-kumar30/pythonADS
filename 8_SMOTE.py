import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Drop unnecessary columns
data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# Convert categorical features into numerical using One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

# Ensure 'EstimatedSalary' is correctly formatted
data['EstimatedSalary'] = pd.to_numeric(data['EstimatedSalary'], errors='coerce')

# Define features (X) and target variable (y)
X = data.drop(columns=['Exited'])  # All columns except 'Exited' (independent variables) Features
y = data['Exited']  # Target variable

# Check class imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=['Not Exited', 'Exited'])
plt.show()
print(y.value_counts())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model without handling imbalance
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the model before applying SMOTE
print("Classification Report (Before SMOTE):\n", classification_report(y_test, y_pred))

# Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset
smote = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42) # minority class will be 50% of the majority class
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=['Not Exited', 'Exited'])
plt.show()
print(y_train_resampled.value_counts())

# Train model after handling imbalance
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_smote = clf_smote.predict(X_test)

# Evaluate the model after SMOTE
print("Classification Report (After SMOTE):\n", classification_report(y_test, y_pred_smote))

# Recall (Exited = 1) increased from 46% → 56% (Now detecting more churn cases).
# Precision (Exited = 1) dropped from 78% → 67% (More false positives, but recall is better).
# Accuracy remains similar (85%), but the model is now better at predicting churn.