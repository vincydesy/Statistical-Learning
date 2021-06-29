import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def normalize(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / (max_value - min_value)
    return to_normalize

def train_test_split(x,y,k):
    # print (data.shape)
    num_of_rows = (k) * 0.8  # inserisco un numero per una certa confidenza

    np.random.shuffle(data)  # prendo valori random
    train_data = x[:int(num_of_rows)]  # creo struttura di train (num of rows)
    test_data = x[int(num_of_rows):]  # creo struttura di test (dataset - num of rows)
    train_labels = y[:int(num_of_rows)]
    test_labels = y[int(num_of_rows):]

    return train_data,test_data,train_labels,test_labels

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
        #temp_accuracy = ac.A_micro_average(curr_test_labels, prediction)
        temp_accuracy = get_accuracy_f1(prediction,curr_test_labels)
        accuracy += temp_accuracy

    accuracy/=k
    return accuracy

def backward_feature_elimination(x,model,y):

    # creo la lista che conterrà accuratezza migliore e dataset modificato, prendo la grandezza di quest'ultimo
    n_features = x.shape[1]
    res = [None] * 2

    while (n_features > 1): # finchè il dataset non sarà svuotato

        iteration=0 #setto il check di continuo/fine a 0
        for i in reversed(range(n_features)): # vado ad eliminare le caratteristiche all'indietro
           modified_x = np.delete(x,i,1)
           accuracy = k_cross_valid(model,modified_x,y,3) # eseguo una validazione ad ogni cancellazione
           if res[0] == None:
               iteration=1
               res[0] = accuracy
               res[1] = modified_x
           else:
               if accuracy>res[0]: # se l'accuratezza è migliore, sostituisco il nuovo dataset nella lista
                   iteration = 1
                   res[0] = accuracy
                   res[1] = modified_x

        if iteration==1: # se ho trovato un dataset migliore con n-1 caratteristiche, continuo, altrimenti esco
            print(res[1].shape)
            x = res[1]
            n_features = x.shape[1]
        else:
            break

    return res

# main

#dataset = load_iris()
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
n_sample = x.shape[0]  # dim dataset righe
n_features = x.shape[1]

for i in range(n_features):
    min = np.min(x[:, i])
    max = np.max(x[:, i])
    data = normalize(x, i, max, min)

x_train, x_test, y_train, y_test = train_test_split(x,y,95) # numero di train dato dall'utente
#model = LogisticRegression(C=1000.0, random_state=0)
model = svm.SVC(kernel="rbf", random_state=0)
bf = backward_feature_elimination(x_train,model,y_train)
new_features=bf[1].shape[1]
print ("Dataset con",new_features,"Features in backward ha accuratezza:", bf[0])
