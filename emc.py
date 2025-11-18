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
