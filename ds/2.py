import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt

# Function to create correlation heatmap
def plot_correlation_heatmap(df):
    """Plot a correlation heatmap of numerical variables"""
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Function to perform chi-square test
def perform_chi_square(df, var1, var2):
    """Perform chi-square test between two categorical variables"""
    # Create contingency table
    contingency_table = pd.crosstab(df[var1], df[var2])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Print results
    print(f"Chi-square test between '{var1}' and '{var2}':")
    print(f"Chi-square value: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Significance: {'Significant' if p_value < 0.05 else 'Not significant'} at 5% level")
    
    # Display contingency table
    print("\nContingency Table:")
    print(contingency_table)

# Example usage
if __name__ == "__main__":
    # Load your dataset (replace with your file path)
    file_path = input("Enter path to your dataset file (CSV or Excel): ")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    # 1. Correlation Heatmap
    plot_correlation_heatmap(df)
    
    # 2. Chi-square Test
    print("\nAvailable categorical columns:")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for i, col in enumerate(cat_cols, 1):
        print(f"{i}. {col}")
    
    if len(cat_cols) >= 2:
        var1_idx = int(input("\nSelect first variable number: ")) - 1
        var2_idx = int(input("Select second variable number: ")) - 1
        perform_chi_square(df, cat_cols[var1_idx], cat_cols[var2_idx])
    else:
        print("Not enough categorical columns for chi-square test.")