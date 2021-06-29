import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.special import comb
from sklearn.datasets import load_iris

def normalize(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / (max_value - min_value)
    return to_normalize

def rand_index(y_clust, y_class):
    tp_plus_fp = comb(np.bincount(y_clust), 2).sum() #somma di coefficiente binomiale (occorrenza,2) per tutte le occorrenze
    tp_plus_fn = comb(np.bincount(y_class), 2).sum() # lo stesso ma per le classi
    A = np.c_[(y_clust, y_class)] # per ogni coppia, traformo i due vettori in una matrice colonna
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(y_clust)) # sommo il coefficiente binomiale su tuto il set delle predizioni
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    R_I = (tp + tn) / (tp + fp + fn + tn)
    return A, R_I

dataset = load_iris()

x = dataset.data
y = dataset.target
n_sample = x.shape[0]  # dim dataset righe
n_features = x.shape[1]

for i in range(n_features):
    min = np.min(x[:, i])
    max = np.max(x[:, i])
    data = normalize(x, i, max, min)

agg_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', compute_full_tree='auto', linkage='average') # 'ward' 'single' 'average' 'complete'
y_clust = agg_cluster.fit_predict(data)
y_class = y
A,rand = rand_index(y_clust,y_class)
print("Rand Index: ", rand)
