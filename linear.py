#9) Linear Regression 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10        # Feature
y = 3 * X.flatten() + 5 + np.random.randn(100) * 2   # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Output results
print("Slope (Coefficient):", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
