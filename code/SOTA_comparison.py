import time
import warnings
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import hdbscan
from sklearn.mixture import BayesianGaussianMixture
np.random.seed(0)

method = 'DPMM'

XObs = np.loadtxt('newSet2.csv')
#XObs = np.loadtxt('T15_track_obs.txt', usecols=(1,2,3,4))
print(XObs.shape)
#XObs = np.c_[x1, y1, x2, y2] # Not including time dimension for MIT & GCS


params = {'quantile': .01, #'quantile': .3, .0549 :smaller value more number of clusters
            'eps': 1.15,# smaller value more number of clusters Earlier value = 0.83
            'damping': 0.9, #'damping': .9,
            'preference': -1.4, #'preference': -200,
            'n_neighbors': 10,
            'n_clusters': 23} #'n_clusters': 3}

#params = default_base.copy()

X = StandardScaler().fit_transform(XObs) #5-D

bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
X, n_neighbors=params['n_neighbors'], include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
X, n_neighbors=params['n_neighbors'], include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

# ============
# Create cluster objects
# ============
#Meanshift
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#DBSCAN
#print 'eps = ',params['eps']
dbscan = cluster.DBSCAN(eps=params['eps'],min_samples=1)

#AP
affinity_propagation = cluster.AffinityPropagation(
damping=params['damping'], preference=params['preference'])

#HDBSCAN
hdb = hdbscan.HDBSCAN(algorithm = 'boruvka_kdtree', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40, metric='euclidean', min_cluster_size=270, min_samples=1, p=None)

#DPMM
dpmm = BayesianGaussianMixture(n_components = 20, weight_concentration_prior = 1, weight_concentration_prior_type='dirichlet_process')


clustering_algorithms = {'AP' : affinity_propagation,
                        'MS' : ms,
                        'DBSCAN' : dbscan,
                        'HDBSCAN' : hdb,
                        'DPMM' : dpmm}


algorithm = clustering_algorithms[method]
t0 = time.time()
    # catch warnings related to kneighbors_graph
with warnings.catch_warnings():
    warnings.filterwarnings(
         "ignore", message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.",
        category=UserWarning)
    warnings.filterwarnings("ignore", message="Graph is not fully connected, spectral embedding may not work as expected.", category=UserWarning)
    algorithm.fit(X)

t1 = time.time()
print (method, (t1 - t0) * 1000)
if hasattr(algorithm, 'labels_'):
    y_pred = algorithm.labels_.astype(np.int)
else:
    y_pred = algorithm.predict(X)

np.savetxt('cluster__'+ method +'.csv',y_pred, fmt='%i')
print(len(np.unique(y_pred)), np.unique(y_pred))
#   np.savetxt(name+'/'+name+'_cluster_result.out', np.c_[tid, x1, y1, x2, y2, t, y_pred], fmt="%i", delimiter='\t')
    

labels, counts = np.unique(y_pred, return_counts=True)
Colors = np.random.rand(len(labels),3)


fig = plt.figure()
for i,c in enumerate(labels):
    x, y = XObs[y_pred == c, 0], XObs[y_pred == c, 1]
    plt.scatter(x, y, marker = '.', color=Colors[i])

plt.show()

