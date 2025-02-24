import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Load dataset
file_path = "loan_data_set.csv"
df = pd.read_csv(file_path)

# Convert all column values to string and remove leading/trailing spaces
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# Replace common missing value indicators with NaN
df.replace(['--', 'NA', 'NaN', 'nan', 'null', 'None', 'N/A', 'n/a'], np.nan, inplace=True)

# Convert numerical columns back to numeric type
df = df.apply(pd.to_numeric, errors='ignore')

# Drop Loan_ID as it's just an identifier
df.drop(columns=['Loan_ID'], inplace=True)

# Visualize missing values
msno.bar(df)
plt.show()

# Select numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Check for problematic values in numeric columns
for col in df_numeric.columns:
    print(f"Unique values in {col}:", df_numeric[col].unique())

# Convert all numeric columns to float
df_numeric = df_numeric.astype(float)

# Heatmap of numeric data
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Check missing values before imputation
missing_values = df.isnull().sum()
print("Missing values before imputation:\n", missing_values)

# Categorical columns with missing values
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Numerical columns with missing values
numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Verify missing values are handled
print("Missing values after imputation:\n", df.isnull().sum())

# Convert categorical columns to numerical using Ordinal Encoding
encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

# Summary statistics
print("Descriptive Statistics:\n", df.describe())

# Boxplot for LoanAmount
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['LoanAmount'])
plt.title("Boxplot of Loan Amount")
plt.show()

# Scatter plot of ApplicantIncome vs LoanAmount
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['ApplicantIncome'], y=df['LoanAmount'])
plt.title("Applicant Income vs Loan Amount")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_loan_data.csv", index=False)
print("Cleaned dataset saved as 'cleaned_loan_data.csv'")
