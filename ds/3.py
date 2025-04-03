import pandas as pd
import numpy as np
import seaborn as sns

# Import necessary libraries
import matplotlib.pyplot as plt

# Read the dataset (replace 'your_dataset.csv' with your file)
df = pd.read_csv('Groceries_dataset.csv')

# Basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Display first few rows
print("\nFirst few rows:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistical description
print("\nStatistical Description:")
print(df.describe())

# Correlation matrix
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['int64', 'float64'])
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
else:
    print("No numeric columns available for correlation analysis")

# Distribution plots for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.show()

# Box plots for numerical columns
plt.figure(figsize=(12, 6))
df.boxplot(column=numerical_cols)
plt.xticks(rotation=45)
plt.title('Box Plots of Numerical Variables')
plt.show()

# Count plots for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(f'Count Plot of {col}')
    plt.show()