import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import trim_mean
import matplotlib.pyplot as plt

# Load dataset
file_path = "loan_data_set.csv"
df = pd.read_csv(file_path)

# Drop Loan_ID as it's just an identifier
df.drop(columns=['Loan_ID'], inplace=True)

# Check missing values
missing_values = df.isnull().sum()
print("Missing values before imputation:\n", missing_values)

# Categorical columns with missing values
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)  # Mode imputation

# Numerical columns with missing values
numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)  # Median imputation
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)  # Mode imputation
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)  # Mode imputation

# Verify missing values are handled
print("Missing values after imputation:\n", df.isnull().sum())

# Convert categorical columns to numerical using Ordinal Encoding
encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

# Summary statistics
desc_stats = df.describe()
print("Descriptive Statistics:\n", desc_stats)

# Boxplot for LoanAmount
df['LoanAmount'].plot(kind='box', vert=False, figsize=(10, 5))
plt.title("Boxplot of Loan Amount")
plt.show()

# Scatter plot of ApplicantIncome vs LoanAmount
plt.scatter(df['ApplicantIncome'], df['LoanAmount'], c='blue', alpha=0.5)
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Applicant Income vs Loan Amount')
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_loan_data.csv", index=False)
print("Cleaned dataset saved as 'cleaned_loan_data.csv'")