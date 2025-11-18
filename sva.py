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
