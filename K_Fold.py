import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn import svm
import random as rndm
import Accuracies as ac

def mean_square_error(predicts_label, real_label):
    diff = np.square(predicts_label - real_label)
    return np.mean(diff)

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
        temp_accuracy = mean_square_error(prediction,curr_test_labels)
        #temp_accuracy = get_accuracy_f1(prediction,curr_test_labels)
        accuracy += temp_accuracy

    accuracy/=k
    return accuracy

x, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
model = svm.SVC(kernel="rbf")
k=8 # iterazione e split

print("Accuratezza: ", k_cross_valid(model,x,y,k))
