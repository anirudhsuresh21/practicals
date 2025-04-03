import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_data.csv' with your actual file path
df = pd.read_csv('your_data.csv')

# Assuming your CSV has columns 'X' and 'y'
X = df[['X']].values  # Independent variable
y = df['y'].values    # Dependent variable

# Rest of the code remains the same
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
correlation = np.corrcoef(y, y_pred)[0, 1]

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Correlation Coefficient: {correlation:.4f}")

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

if r2 >= 0.8:
    print("Good fit: Model explains more than 80% of the variance in the data")
elif r2 >= 0.6:
    print("Moderate fit: Model explains 60-80% of the variance")
else:
    print("Poor fit: Model explains less than 60% of the variance")
