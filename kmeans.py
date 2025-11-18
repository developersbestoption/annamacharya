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
