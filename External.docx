# 2. PREPROCESSING TECHNIQUES
#a) Attribute Selection
import pandas as pd

df = pd.DataFrame({
    "id": [1,2,3],
    "name": ["Ravi","Anjali","Vikram"],
    "marks": [95, 88, 76],
    "grade": ["A+","A","B"]
})
print("original dataset:\n",df)
# Select important attributes
selected = df[["name", "marks"]]
print("\n",selected)
print()

#b) Handling Missing Values (Mean, Median, Mode)
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["Ravi","Anjali","Vikram"],
    "marks": [95, np.nan, 76]
})

df["marks_mean"] = df["marks"].fillna(df["marks"].mean())
df["marks_median"] = df["marks"].fillna(df["marks"].median())
df["marks_mode"] = df["marks"].fillna(df["marks"].mode()[0])

print(df)
print()

#C) Discretization
import pandas as pd

df = pd.DataFrame({
    "marks": [95, 88, 76, 90]
})

# 3 bins → 3 labels
df["bin"] = pd.cut(df["marks"], bins=3, labels=["Low", "Medium", "High"])
print(df)
print()

#D) Outlier Elimination (IQR Method)
import pandas as pd

df = pd.DataFrame({
    "marks": [95, 88, 76, 800]
})

q1 = df["marks"].quantile(0.25)
q3 = df["marks"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

filtered = df[(df["marks"] >= lower) & (df["marks"] <= upper)]
print(filtered)
print()
#3) KNN Algorithm

# KNN using scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
# Example predict for a single sample:
print("Predicted class for sample 0:", knn.predict(X_test[:1])[0])
#4) Decision Tree Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create model
dt = DecisionTreeClassifier(random_state=42)

# Train
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
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
#6) Random Forest classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Accuracy
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
#7) Navie bayes classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (features + labels)
X = [[1, 20], [2, 21], [3, 22], [8, 30], [9, 31], [10, 32]]  # features
y = [0, 0, 0, 1, 1, 1]  # class labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create model
model = GaussianNB()

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

print("Predictions:", pred)
print("Accuracy:", accuracy_score(y_test,pred))
#8) support vector algorithm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# SVM Model (linear kernel)
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Accuracy
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

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
#10 logistic regression 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Use only 2 classes for binary classification
iris = load_iris()
X = iris.data
y = (iris.target != 0).astype(int)   # convert to 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
#11) Multi layer perception
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# MLP model (1 hidden layer with 10 neurons)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
mlp.fit(X_train, y_train)

# Predict
y_pred = mlp.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, y_pred))
#12) K Means algorithm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create sample 2D data
np.random.seed(0)
X = np.vstack([
    np.random.randn(40, 2) + [2, 2],
    np.random.randn(40, 2) + [-2, -1],
    np.random.randn(40, 2) + [4, -3]
])

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Plot
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X')
plt.title("K-Means Clustering")
plt.show()
#13) Fuzzy C Means clustering

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Sample data
np.random.seed(0)
X = np.vstack([
    np.random.randn(40, 2) + [2, 2],
    np.random.randn(40, 2) + [-2, -1],
    np.random.randn(40, 2) + [4, -3]
]).T  # FCM needs data in shape (features, samples)

# Apply Fuzzy C-Means
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X, c=3, m=2, error=0.005, maxiter=1000)

# Most likely cluster labels
labels = np.argmax(u, axis=0)

# Plot
plt.scatter(X[0], X[1], c=labels)
plt.scatter(cntr[:,0], cntr[:,1], c='red', s=150, marker='X')
plt.title("Fuzzy C-Means Clustering")
plt.show()
#14) Exception maximizing clustering algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(gmm.means_[:,0], gmm.means_[:,1], marker='X', s=200, c='red')
plt.title("EM (Gaussian Mixture Model)")
plt.show()
