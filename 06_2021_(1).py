"Si consideri il data set Iris e si applichi un algoritmo di clustering gerarchico."
"Si scelga il miglior valore di K"
"Librerie concesse: moduli numpy, le funzioni fit e predict di sklearn per il modello considerato, sklearn.dataset.load_<dataset>, sklearn.cluster e, se necessario, matplotlib.pyplot."

from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import AgglomerativeClustering

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

    # calcolo la purezza, in ogni subcluster conto le occorrenze dell'elemento più frequente, le sommo e divido il tutto per la lunghezza totale
    purity = 0
    for c in clusters_labels:
        if type(c) is np.ndarray:
            y = np.bincount(c)  # trovo occorrenze degli elementi presenti
            maximum = np.max(y)  # prendo l'elemento con maggiore frequenza
            purity += maximum
        else: # nel caso di un solo elemento
            purity += 1

    purity = purity/len_clust

    return purity

def best_HAC(X, Y, k):
    # testo le varie affinità studiate con i vari tipi di linkage
    affinità = ['euclidean', 'manhattan', 'cosine']
    linkage = ['complete', 'average', 'single']
    grid_linkage = []

    for l in linkage:
        for a in affinità:
            grid_linkage.append([l, a])

    grid_linkage.append(['ward', 'euclidean'])  # ward funziona solo con la distanza euclidea
    best_purity = 0
    best_parameter = None

    for i in range(2, k):
        for p in grid_linkage: # cerco i parametri migliori per agglomerative clustering
            model = AgglomerativeClustering(n_clusters=i, affinity=p[1], linkage=p[0])
            cluster_pred = model.fit_predict(X)
            purity_score = purity(cluster_pred,Y)
            if best_parameter is None or purity_score > best_purity:
                best_purity = purity_score
                best_parameter = p
                best_k = i

    return best_purity,best_parameter,best_k

# inizio script

dataset = load_iris()
data, target = dataset.data, dataset.target
purity_score, parameter, k = best_HAC(data, target, 10)
print("Purezza:", purity_score)
print("Tipo di HAC:",parameter)
print("Miglior K:",k)


