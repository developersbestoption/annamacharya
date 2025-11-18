#5) Decision Tree Regression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Sample dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.flatten() + np.random.randn(100) * 2

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Model
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# MSE
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
