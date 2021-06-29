import numpy as np
import random as rndm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

"Si valutino le prestazione del dataset Load Breast Cancer Wisconsin con Logistic Regression"

def confusion_matrix_multiclass(labels, predictions):  # 00=TP  10=FN  01=FP  11=TN
    n_labels = np.max(labels) + 1  # numero di classi
    sample_size = labels.shape[0]  # numero istanze classification

    # inizializzo i parametri
    FP = FN = TP = TN = 0

    for i in range(n_labels):  # PER OGNI CLASSE
        for j in range(sample_size):  # calcolo i parametri

            if labels[j] == i:
                if predictions[j] == i:
                    TP += 1
                else:
                    FN += 1
            else:
                if predictions[j] != i:
                    TN += 1
                else:
                    FP += 1

    cnf_matrix = np.zeros((2, 2))
    cnf_matrix[0, 0] = TP
    cnf_matrix[1, 0] = FN
    cnf_matrix[0, 1] = FP
    cnf_matrix[1, 1] = TN

    return cnf_matrix

def A_micro_average(labels, predictions):
    matrix = confusion_matrix_multiclass(labels, predictions)
    TP = matrix[0, 0]
    FN = matrix[1, 0]
    FP = matrix[0, 1]
    TN = matrix[1, 1]

    A = (TP + TN) / (TP + TN + FP + FN)
    return A

def normalize(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / (max_value - min_value)
    return to_normalize

def k_split(x,k):
    x_size = x.shape[0]
    fold_size = int(x_size/k)
    folds_size = [fold_size] * k # creo k liste che conterranno il dataset diviso in k parti
    folds = np.array(range(k))
    indices = np.zeros(x_size) # creo un array degli indici di dimensione del dataset

    for i in range(x_size):  # per dimensione del dataset
        index = rndm.choice(folds) # scelgo a caso un fold da prendere (k lungo)
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
        temp_accuracy = A_micro_average(prediction,curr_test_labels)
        accuracy += temp_accuracy

    accuracy/=k
    return accuracy

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
n_features = x.shape[1]

for i in range(n_features):
    min = np.min(x[:, i])
    max = np.max(x[:, i])
    data = normalize(x, i, max, min)

model = LogisticRegression()
k = 8  # iterazione e split

print("Accuratezza: ", k_cross_valid(model, data, y, k))