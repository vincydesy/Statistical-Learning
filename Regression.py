import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def normalize_transform(to_normalize, column_index, max_value, min_value):
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

            if test_labels[j] >= i:
                if prediction_index[j] >= i:
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


x, y = datasets.load_diabetes(return_X_y=True)
x = x[:, np.newaxis, 2]
x=x[:-20]
y=y[:-20]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)  #70% dati 30% test

normalized_train = x_train
normalized_test = x_test

iteration=x.shape[1] #normalizzazione (non serve in questo caso)
for i in range(iteration):
    max = np.max(x_train[:, i])
    min = np.min(x_train[:, i])
    normalized_train = normalize_transform(normalized_train, i, max, min)
    normalized_test = normalize_transform(normalized_test, i, max, min)
x_com_std = np.vstack((normalized_train,normalized_test))
y_com_std = np.hstack((y_train,y_test))

#regressione logistica su 100 iterazioni
log_model = LogisticRegression(C=100.0, random_state=0)
log_model.fit(x_train, y_train)
result = log_model.predict(x_test)
plt.scatter(x_train,y_train)
plt.show()
plt.scatter(result,y_test)
plt.show()
accuracy_log = get_accuracy_f1(result,y_test)
print("Accuratezza con reg. logistica: ",accuracy_log)

# regressione lineare (non ottimale su questo dataset)
lin_model = LinearRegression()
lin_model.fit(x_train, y_train)
lin_result = lin_model.predict(x_test)
plt.scatter(lin_result,y_test)
plt.show()
accuracy_lin= get_accuracy_f1(lin_result,y_test)
print("Accuratezza con reg. lineare: ",accuracy_lin)