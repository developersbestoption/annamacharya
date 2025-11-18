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
