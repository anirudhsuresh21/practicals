import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

def preprocess_data(df):
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Identify categorical columns (object type)
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    return df

def analyze_dataset(file_path, target_column, feature_columns=None):
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # If feature columns not specified, use all columns except target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Split features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Naive Bayes
    print("\nNaive Bayes Results:")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, nb_pred) * 100:.2f}%')
    
    # PCA
    print("\nPCA Results:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=min(4, len(feature_columns)))
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    pca_model = LogisticRegression(random_state=42)
    pca_model.fit(X_train_pca, y_train)
    pca_pred = pca_model.predict(X_test_pca)
    print(f"Accuracy with PCA: {accuracy_score(y_test, pca_pred):.2f}")
    
    # Linear Regression
    print("\nLinear Regression Results:")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    mse = mean_squared_error(y_test, lr_pred)
    mae = mean_absolute_error(y_test, lr_pred)
    r2 = r2_score(y_test, lr_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    if len(feature_columns) >= 2:
        sns.pairplot(df.sample(n=min(100, len(df))), 
                    x_vars=feature_columns[:2], 
                    y_vars=target_column, 
                    height=5, aspect=0.8, kind='reg')
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Specify dataset and parameters for diabetes analysis
    file_path = 'diabetes.csv'
    target_column = 'Outcome'
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    analyze_dataset(file_path, target_column, feature_columns)
