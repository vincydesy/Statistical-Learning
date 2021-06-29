import numpy as np
from sklearn import svm
from sklearn.datasets import load_breast_cancer
import random

"Si prendano le migliori 10 features del dataset load_breast_cancer"

def normalize(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / (max_value - min_value)
    return to_normalize

def get_accuracy_f1(prediction_index, test_labels):
    n_labels = int(np.max(test_labels) + 1)
    set_size = test_labels.shape[0]

    # inizializzo i parametri
    FP = 0
    FN = 0
    TP = 0
    TN = 0

    for i in range(n_labels):
        for j in range(set_size):

            if test_labels[j] == i:
                if prediction_index[j] == i:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction_index[j] != i:
                    TN += 1
                else:
                    FP += 1

    P = TP / (TP + FP)  # precision
    R = TP / (TP + FN)  # recall

    F1 = (2 * P * R) / (P + R)  # f1 score
    return F1


def k_split(x,k):
    x_size = x.shape[0]
    fold_size = int(x_size/k)
    folds_size = [fold_size] * k # creo k liste che conterranno il dataset diviso in k parti
    folds = np.array(range(k))
    indices = np.zeros(x_size) # creo un array degli indici di dimensione del dataset

    for i in range(x_size):  # per dimensione del dataset
        index = random.choice(folds) # scelgo a caso un fold da prendere (k lungo)
        folds_size[index] -= 1 # scelgo la lista con indice precedente a quella scelta
        if folds_size[index] == 0 and i == x_size - 1: # prendo i fold differenti da quelli scelti
            folds = folds[folds != index]
        indices[i] = index

    return indices

def k_cross_valid(model,x,y,k):
    x_size = x.shape[0]
    splits = k_split(x,k)
    accuracy=0

    for i in range(k): # azzero gli array di test e train
        curr_train = None
        curr_test = None
        curr_train_labels = []
        curr_test_labels = []

        for j in range(x_size):
            if splits[j] == i: #se trovo uno split allora lo inserisco nelle labels test
                curr_test_labels.append(y[j])
                if curr_test is None: # inserimento nel test set
                    curr_test = x[j]
                else:
                    curr_test = np.vstack((curr_test, x[j]))
            else: # lo inserisco altrimenti nelle labels train
                curr_train_labels.append(y[j])
                if curr_train is None: # inserimento nel train set
                    curr_train = x[j]
                else:
                    curr_train = np.vstack((curr_train, x[j]))

        curr_train_labels = np.array(curr_train_labels)
        curr_test_labels = np.array(curr_test_labels)
        model.fit(curr_train, curr_train_labels)
        prediction = model.predict(curr_test)
        #temp_accuracy = ac.A_macro_average(curr_test_labels, prediction)
        temp_accuracy = get_accuracy_f1(prediction,curr_test_labels)
        accuracy += temp_accuracy

    accuracy/=k
    return accuracy


def forward_feature_elimination(x,model,y,k):

    # creo la lista che conterrà accuratezza migliore e dataset modificato, prendo la grandezza di quest'ultimo
    res = [None] * 2
    res[0] = 0 # setto a 0 l'accuratezza iniziale
    n_features = x.shape[1]
    best_features = n_features-k # features richieste dall'utente
    iteration = 0

    while (n_features > best_features): # finchè il dataset non avrà n features

        if iteration == 0: # passo base, inserisco la prima feature
            for i in range(n_features): # vado ad inserire la caratteristica una alla volta
                modified_x = x[:,i]
                accuracy = k_cross_valid(model,modified_x,y,5) # eseguo una validazione ad ogni inserimento
                if accuracy>res[0]: # se l'accuratezza è migliore, sostituisco il nuovo dataset nella lista
                    res[0] = accuracy
                    res[1] = modified_x
                    last_best = i
            iteration = 1 # setto il flag ad 1 per il passo induttivo
        else: # passo induttivo, creo una lista temporanea che mi conterrà il dataset con n+1 features
            temp = [None] * 2
            temp[0] = 0 # setto l'accuratezza temporanea
            for i in range(n_features): # vado ad inserire la caratteristica una alla volta
                modified_x = res[1]
                modified_x = np.column_stack((modified_x, x[:,i]))
                accuracy = k_cross_valid(model,modified_x,y,5) # eseguo una validazione ad ogni inserimento
                if accuracy>temp[0]: # se l'accuratezza è migliore, sostituisco il nuovo dataset nella lista
                    temp[0] = accuracy
                    temp[1] = modified_x
                    last_best = i
            res = temp


        x = np.delete(x,last_best,1)
        n_features = x.shape[1]


    return res

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
n_features = x.shape[1]

for i in range(n_features):
    min = np.min(x[:, i])
    max = np.max(x[:, i])
    data = normalize(x, i, max, min)


model = svm.SVC(kernel="linear")
ff = forward_feature_elimination(data,model,y,10)
new_features=ff[1].shape[1]
print ("Dataset con",new_features,"Features in forward ha accuratezza:", ff[0])


