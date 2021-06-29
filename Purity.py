import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris

def normalize(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / (max_value - min_value)
    return to_normalize

def purity(y_clust,y_class):

    #setto lunghezze e creo la lista di subclusters
    size_clust = np.max(y_clust)+1
    len_clust = len(y_clust)
    clusters_labels = [None] * size_clust

    # su tutto il cluster, per ogni elemento, aggiungo la classe nel suo subcluster
    for i in range(len_clust):
        index = y_clust[i]
        if clusters_labels[index] is None:
            clusters_labels[index] = y_class[i]
        else:
            clusters_labels[index] = np.hstack((clusters_labels[index], y_class[i]))

    #calcolo la purezza, in ogni subclster conto le occorrenze dell'elemento più frequente, le sommo e divido il tutto per la lunghezza totale
    purity = 0
    for c in clusters_labels:
        #print(c)
        y = np.bincount(c) #trovo occorrenze degli elementi presenti
        #print(y)
        maximum = np.max(y) # prendo l'elemento con maggiore frequenza
        purity += maximum

    purity = purity/len_clust

    return purity



#x, y = make_blobs(n_samples=15000, random_state=170) #creo un cluster di n dimensione e periodo numerico generico
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
print("Purezza: ", purity(y_clust,y_class))