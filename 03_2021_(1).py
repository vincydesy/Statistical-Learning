#Si consideri il data set digits e si risponda alla seguente domanda: quale tra le possibili trasformazioni delle features viste al corso funziona meglio rispetto alle classi del data set con un algoritmo di clustering k-means?"
"Bocciato perchè ho usato doppia trasformazione (boh, calano di molto le prestazioni)"

from sklearn.datasets import load_digits
import numpy as np
import random as rndm

def normalize_transform(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / ((max_value - min_value)+0.1)
    return to_normalize


def sig_transform(to_normalize, column_index, b):
    to_normalize[:, column_index] = 1 / (1 + np.exp(b * to_normalize[:, column_index]))
    return to_normalize


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
        index = euc_dist.index(np.min(euc_dist))
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

    return clusters,centroidi


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


# inizio script

dataset = load_digits()
x,y = dataset.data, dataset.target
clusters,centroidi = k_means(x,8,8) # divido in dati in 8 clusters su 8 iterazioni
print("RSS prima delle trasformazioni:",rss(clusters))
to_delete = list() # lista delle colonne da eliminare

i=0
for f in range(x.shape[1]): # elimino le feature con tutti gli elementi in colonna uguali poichè poco importanti al fine della classificazione
        val= None
        all_equal = True
        for k in range(x.shape[0]):
            if val is None:
                val = x[k,f]
            elif x[k,f] != val:
                all_equal = False
                break
        if all_equal:
            to_delete.append(f-i)
            i+=1

for l in to_delete:
    x = np.delete(x,l,1)

n_features = x.shape[1]
for i in range(n_features): # effettuo una doppia trasformazione per diminuire la complessità del dataset
    min = np.min(x[:, i])
    max = np.max(x[:, i])
    data = normalize_transform(x, i, max, min) # normalizzo il dataset per approssimare eventuali errori di stima
    data = sig_transform(x,i,1) # atttraverso la sigmoide, data l'elevata discrepanza tra valori nulli e quelli alti, effettuo un "ritaglio morbido" sui valori alti

clusters,centroidi = k_means(x,8,8) # divido in dati in 8 clusters su 8 iterazioni
print("RSS dopo le trasformazioni:",rss(clusters)) # eseguo un approccio senza supervisione per le prestazioni
print("Dopo eventuali strategie provate, la doppia trasformazione normalizzazione-sigmoide si è rivelata la più efficiente, in quanto queste hanno permesso di ridurre la discrepanza tra i valori troppo bassi con quelli molto alti")
