import numpy as np
import random as rndm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import KMeans



def euclidean_distance(x, y): #distanza euclidea
    sum_sq = np.sum(np.square(x - y))
    return np.sqrt(sum_sq)


def recalculate_clusters(X, centroids, k):
    # Creo una lista di k array subclusters
    clusters = [None] * k
    for data in X:
        # Creo un array delle distanze euclidee
        euc_dist = []
        # per ogni punto sul dataset calcolo la distanza euclidea rispetto ai k centroidi generati e la aggiungo all'array
        for j in range(k):
            euc_dist.append(euclidean_distance(data,centroids[j]))
        # Aggiungo il punto al k subcluster con la distanza minore tenendomi l'indice (sempre k) della distanza euclidea
        index = euc_dist.index(min(euc_dist))
        if clusters[index] is None:
            clusters[index] = data;
        else:
            clusters[index] = np.vstack((clusters[index], data))
    return clusters

def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        # Per ogni subcluster ricalcolo il centroide facendo la media di tutti i punti
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids

def convert_list_clusters_toarray(clusters):

    # converto la lista dei subcluster in un array totale unico senza distinzioni

    clusters_array = (None)
    for i in range(len(clusters)):
        if clusters_array is None:
            clusters_array = np.array(clusters[i])
        else:
            clusters_array = np.concatenate((clusters_array, clusters[i]))

    return clusters_array


def k_means(x,k,n_iter):

    #setto le dimensioni dell'array di centroidi e del totale del dataset
    set_dim = x.shape[0]
    centroidi = [None] * k

    # setto centroidi a caso
    indicicentroidi = rndm.sample(range(0, set_dim + 1), k)
    for i in range(k):
        centroidi[i] = x[indicicentroidi[i]]

    # per ogni iterazione richiesta dall'utente, ricalcolo subclusters e centroidi
    for iter in range(n_iter):
        clusters = recalculate_clusters(x,centroidi,k) #ritorna una lista di array di dimensione k contenente i k subclusters
        centroidi = recalculate_centroids(centroidi, clusters, k)

    # converto la lista di centroidi in un array per plottarlo
    centroidi = np.array(centroidi)
    #clusters = convert_list_clusters_toarray(clusters)

    return clusters,centroidi

x, y = make_blobs(n_samples=15000, random_state=170) #creo un cluster di n dimensione e periodo numerico generico
k = np.max(y)+1
iterations = 8
kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
clusters,centroidi = k_means(x,k,iterations)
colors = cycle('bgrmk') #setto l'array di colori (serve libreria che sta sopra)
for c in clusters: #scorro la lista così da stampare un colore diverso per ogni subcluster
    plt.scatter(c[:,0],c[:,1], c=next(colors))
plt.scatter(centroidi[:,0],centroidi[:,1], c=next(colors))
plt.show()



