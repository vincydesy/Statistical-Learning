import numpy as np
import random as rndm
from sklearn.datasets import load_diabetes

"Si prenda il dataset diabetes, si rendano i dati etichettati e valutare le prestazioni di questi"

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


def rss(clust):

    RSS = 0
    for c in clust:
        mu = centroide(c)  # calcola centroide del subcluster c
        RSS += distanza_centroide(c, mu)  # calcola l'RSS del cluster totale per ogni c subcluster

    return RSS


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


##  main

dataset = load_diabetes()
x = dataset.data
k = 3
iterations = 8
clusters,centroidi = k_means(x,k,iterations)
rss = rss(clusters)
print("RSS :" ,rss)


