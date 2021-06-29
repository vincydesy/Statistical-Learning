" Nella soluzione dell'esercizio si possono utilizzare solo i moduli numpy, sklearn e, se necessario, matplotlib.pyplot."
"Si seguano con attenzione le indicazioni date nel testo."
"L'implementazione va commentata non rifrasando l'istruzione di codice, ma spiegandone le motivazioni teoriche."
"Si applichi al dataset iris un algoritmo di clustering di vostra scelta con K=2, 3, e 4."
"Se ne valutino le prestazioni implementando la misura scelta senza usare nessuna libreria."
"Cosa posso concludere da queste misure?"

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

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
        if type(c) is np.ndarray: # nel caso con k=4 il quarto cluster è 0, quindi ho dovuto eseguire il controllo se fosse un numpy array
            #print(c)
            y = np.bincount(c) #trovo occorrenze degli elementi presenti
            #print(y)
            maximum = np.max(y) # prendo l'elemento con maggiore frequenza
            purity += maximum
        else:
            purity += c

    purity = purity/len_clust

    return purity

#  main

dataset = load_iris()
x,y = dataset.data, dataset.target
y_class = y
n_features = x.shape[1]

for i in range(n_features):
    min = np.min(x[:, i])
    max = np.max(x[:, i])
    data = normalize(x, i, max, min)

for i in range(2,5):
    model = AgglomerativeClustering(n_clusters=i, affinity='euclidean', compute_full_tree='auto',
                                          linkage='average')  # 'ward' 'single' 'average' 'complete'
    y_clust = model.fit_predict(data)
    purezza = purity(y_clust, y_class)
    print("Purezza con", i,"cluster:", purezza)