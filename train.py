import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data3.csv")
print("Dataset loaded successfully.")

# Split data into features and target variable
X = df.drop(columns=['medv'])
y = df['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Initialize variables to track the best model
best_model = None
best_model_name = None
best_mse = float('inf')

# Linear Regression Model
print("Training Linear Regression model...")
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression MSE: {mse}, R²: {r2}")

# Plot Actual vs Predicted for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", label="Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Ideal Fit")
plt.xlabel('Actual medv')
plt.ylabel('Predicted medv')
plt.title('Actual vs Predicted medv (Linear Regression)')
plt.legend()
plt.savefig("linear_regression_comparison.png")
plt.clf()
print("Linear Regression plot saved as 'linear_regression_comparison.png'.")

# Update best model if necessary
if mse < best_mse:
    best_mse = mse
    best_model = reg
    best_model_name = "LinearRegressionModel"

# Random Forest Model
print("Training Random Forest model...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest MSE: {mse_rf}, R²: {r2_rf}")

# Plot Actual vs Predicted for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, color="green", label="Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Ideal Fit")
plt.xlabel('Actual medv')
plt.ylabel('Predicted medv')
plt.title('Actual vs Predicted medv (Random Forest)')
plt.legend()
plt.savefig("random_forest_comparison.png")
plt.clf()
print("Random Forest plot saved as 'random_forest_comparison.png'.")

# Update best model if necessary
if mse_rf < best_mse:
    best_mse = mse_rf
    best_model = rf_reg
    best_model_name = "RandomForestModel"
