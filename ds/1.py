import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

def preprocess_data(df):
    """
    Function to preprocess dataset with common cleaning operations
    Args:
        df: Input pandas DataFrame
    Returns:
        Cleaned DataFrame
    """
    # 1. Handling Missing Values
    print("\n--- Missing Values Analysis ---")
    print(df.isnull().sum())
    
    # Fill numeric columns with mean
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        mode_value = df[column].mode()
        if not mode_value.empty:  # Check if mode is not empty
            df[column] = df[column].fillna(mode_value.iloc[0])
        else:
            print(f"Column '{column}' has no mode (all values may be NaN).")

    # 2. Removing Duplicates
    print("\n--- Duplicate Records ---")
    print("Duplicates found:", df.duplicated().sum())
    df = df.drop_duplicates()

    # 3. Handling Noisy Data
    print("\n--- Noisy Data Analysis ---")
    
    # Box Plot for numeric columns
    plt.figure(figsize=(10, 6))
    df[numeric_columns].boxplot()
    plt.title("Box Plot for Numeric Columns")
    plt.xticks(rotation=45)
    plt.show()

    # Remove outliers using IQR method
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Binning example for numeric columns
    for column in numeric_columns:
        if len(df[column].unique()) > 4:  # Only bin if there are enough unique values
            try:
                df[f'{column}_binned'] = pd.qcut(df[column], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            except ValueError:
                print(f"Could not create bins for {column} due to distribution of values")

    print("\n--- Final Dataset Shape ---")
    print(df.shape)
    
    return df

# Example usage:
df = pd.read_csv('titanic.csv')
cleaned_df = preprocess_data(df)