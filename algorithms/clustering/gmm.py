"""Custom gaussian mixture model as practice.

Implement GMM with EM.
https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
https://mml-book.github.io/book/chapter11.pdf
"""

import scipy
import scipy.stats

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


mu_1 = [7, 5]
sigma_1 = [0.6, 0.6]  # 0.36, 0.36
cluster_1 = np.random.normal(mu_1, sigma_1, size=[40, 2])


mu_2 = [3, 4]
sigma_2 = [0.8, 0.6]  # 0.64, 0.36
cluster_2 = np.random.normal(mu_2, sigma_2, size=[30, 2])


mu_3 = [4, 6]
sigma_3 = [0.5, 0.3]  # 0.25, 0.09
cluster_3 = np.random.normal(mu_3, sigma_3, size=[30, 2])

sns.scatterplot(cluster_1[:, 0], cluster_1[:, 1])
sns.scatterplot(cluster_2[:, 0], cluster_2[:, 1])
sns.scatterplot(cluster_3[:, 0], cluster_3[:, 1])
plt.xlim(0, 10)
plt.ylim(0, 10)
# plt.show()

data = np.vstack([
                 cluster_1,
                 cluster_2,
                 cluster_3
                 ])

n_components = 3



class GMM:
    def __init__(self, data, n_components, n_iters=1e2):
        self.data = data
        self.n_components = n_components
        self.n_iters = int(n_iters)

    def em(self):
        # Initialize the cluster means to random points in the data set. Initialize cluster variances.
        indices = self.data.shape[0]
        self.cluster_means = data[np.random.choice(indices, self.n_components, replace=False)]
        self.cluster_variances = np.array([np.cov(np.random.random(self.data.shape), rowvar=False) 
                                           for _ in range(self.n_components)])

        # Initialize some mixture weights
        self.mixture_weights = np.random.random(self.n_components)
        self.mixture_weights = self.mixture_weights / self.mixture_weights.sum()
        assert np.allclose(self.mixture_weights.sum(), 1)


        # Repeat EM algorithm for n_iterations
        for i in range(self.n_iters):
            responsibilities = self.transform(self.data)
            self._e_step(responsibilities)
            self._m_step(responsibilities)


    def transform(self, X):
        assert X.shape[1] == self.data.shape[1]

        responsibilities = np.zeros((X.shape[0], self.n_components))
        
        # Note that these probabilites are in fact probability densities. 
        for i in range(X.shape[0]):
            x = X[i]

            responsibilities[i] = self.mixture_weights * [scipy.stats.multivariate_normal.pdf(
                                                            x, 
                                                            self.cluster_means[k], 
                                                            self.cluster_variances[k]
                                                            ) for k in range(self.n_components)]

        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

        return responsibilities


    def _e_step(self, responsibilities): 
        # Caculate mixture weights by averaging the responsibilities 
        self.mixture_weights = responsibilities.sum(axis=0) / self.data.shape[0]

        return responsibilities

    def _m_step(self, responsibilities):
        # Calculate the responsibilities
        N = responsibilities.sum(axis=0)
        
        for k in range(self.n_components):
            self.cluster_means[k] = (responsibilities[:, k, None] * self.data).sum(axis=0) / N[k]

            Z = self.data - self.cluster_means[k]
            self.cluster_variances[k] = Z.T.dot(responsibilities[:, k, None] * Z) / N[k]



gmm = GMM(data, n_components)
gmm.em()


plt.show()

responsibilities = gmm.transform(data)

labels = np.argmax(responsibilities, axis=1)
sns.scatterplot(data[:,0], data[:, 1], hue=labels)
plt.show()

from sklearn.mixture import GaussianMixture
sk_gmm = GaussianMixture(3)
sk_gmm.fit(data)
sk_responsibilities = sk_gmm.predict_proba(data)
