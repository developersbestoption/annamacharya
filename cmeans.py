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
