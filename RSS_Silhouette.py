import numpy as np
import random as rndm
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_breast_cancer



def centroide(x):
    num_clusters = x.shape[0]
    features = x.shape[1]
    total = np.zeros(features)

    for x in x:
        total = total + x
    total = total / num_clusters

    return total


def distanza_centroide(x, mu):
    tot = 0  # distanza totale

    for x in x:
        tot += np.sum(np.square(np.abs(x - mu)))
    return tot


def rss(x, y, k):
    dimension_total = x.shape[0]
    clust = [None] * k  # ho k subclusters

    for i in range(dimension_total):  # per tutto il cluster, aggiungo ai k subclusters il punto con etichetta n
        if (clust[int(y[i])]) is None:
            clust[int(y[i])] = x[i]
        else:
            clust[int(y[i])] = np.vstack((clust[int(y[i])], x[i]))

    RSS = 0
    for c in clust:
        mu = centroide(c)  # calcola centroide del subcluster c
        RSS += distanza_centroide(c, mu)  # calcola l'RSS del cluster totale per ogni c subcluster

    return RSS


def euclidean_distance(x, y):  # distanza euclidea
    sum_sq = np.sum(np.square(x - y))
    return np.sqrt(sum_sq)


def calcola_coesione(x, c): # su ogni punto, sommo le distanze euclidee tra il punto x e tutti gli altri dello stesso cluster
    w = c.shape[0]
    sum = 0
    for y in c:
        sum = sum + euclidean_distance(x, y)
    total = (1 / (w - 1)) * sum
    return total


def calcola_separazione(x, mycluster, clust): # per ogni punto, prendo la separazione minima, ovvero la somma delle distanze tra x e i punti del cluster più vicino (escluso quello di appartenenza)

    w = 0
    separation = 0
    sum = 0

    for c in clust:
        are_equal = np.array_equal(c,mycluster)
        if not are_equal:
            for y in c:
                sum = sum + euclidean_distance(x, y)
        if separation == 0:
            w = c.shape[0]
            separation = sum
        else:
            if sum < separation:
                separation = sum
                w = c.shape[0]

    total = (1 / w) * separation

    return total


def silhouette(x, y, k):

    dimension_total = x.shape[0]
    clust = [None] * k  # ho k subclusters

    for i in range(dimension_total):  # per tutto il cluster, aggiungo ai k subclusters il punto con etichetta n
        if (clust[int(y[i])]) is None:
            clust[int(y[i])] = x[i]
        else:
            clust[int(y[i])] = np.vstack((clust[int(y[i])], x[i]))

    silhouette_total = 0

    for c in clust:  # su tutti i k clusters
        for x in c:  # su ogni punto dei k cluster
            coesione = calcola_coesione(x, c)
            separazione = calcola_separazione(x, c, clust)
            if coesione >= separazione:
                silhouette_point = (separazione - coesione) / coesione
            else:
                silhouette_point = (separazione - coesione) / separazione # silhouette su ogni punto
            silhouette_total = silhouette_total + silhouette_point

    silhouette_total = silhouette_total / dimension_total

    return silhouette_total


# main
#dataset = load_iris()
dataset = load_breast_cancer()
x = dataset.data
agg_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', compute_full_tree='auto', linkage='average') # 'ward' 'single' 'average' 'complete'
y = agg_cluster.fit_predict(x)
#x, y = make_blobs(n_samples=100, random_state=170)  # creo un cluster di n dimensione e periodo numerico generico
# plt.scatter(x=x[:, 0], y=x[:, 1])  # mostra blob
# plt.show()
k = np.max(y) + 1  # guardo quanti subclusters devo creare sulle etichette
# RSS = RSS(x,y,k)
sil = silhouette(x, y, k)
print(sil)
print(silhouette_score(x, y))