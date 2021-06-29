import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

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
cancer = datasets.load_breast_cancer()
data_features = cancer.feature_names #nomi features
data_labels = cancer.target_names #nomi etichette
data = cancer.data
target = cancer.target
specific_data = data[:,[2, 3, 4, 5, 9, 12, 13, 14, 15, 19, 22, 23, 24, 25, 29]]

x_train, x_test, y_train, y_test = train_test_split(specific_data, target, test_size=0.3,random_state=109) # 70% training and 30% test

model = svm.SVC(kernel="linear")
model.fit(x_train,y_train)
results = model.predict(x_test)
print("Linear Accuracy:",get_accuracy_f1(results,y_test))

model = svm.SVC(kernel="poly", degree=3)
model.fit(x_train,y_train)
results = model.predict(x_test)
print("Poly Accuracy:",get_accuracy_f1(results,y_test))

model = svm.SVC(kernel="rbf")
model.fit(x_train,y_train)
results = model.predict(x_test)
print("RBF Accuracy:",get_accuracy_f1(results,y_test))


