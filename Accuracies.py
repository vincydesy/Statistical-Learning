import numpy as np


def confusion_matrix(labels, predictions):  # 00=TP  10=FN  01=FP  11=TN
    n_sample = labels.shape[0]
    cnf_matrix = np.zeros((2, 2))

    for i in range(n_sample):  # per ogni istanza nel test set,
        if labels[i] > 0:  # se l'etichetta è 1 e se la predizione è 1 incrementa i tp
            if predictions[i] > 0:
                cnf_matrix[0, 0] += 1
            else:
                cnf_matrix[1, 0] += 1  # se la predizione è -1 incrementa i fn
        else:  # se l'etichetta è -1
            if predictions[i] > 0:
                cnf_matrix[1, 1] += 1  # se la predizione è -1 incrementa i tn
            else:
                cnf_matrix[0, 1] += 1  # se la predizione è 1 incrementa i fp
    # calcola l'accuratezza e la ritorna
    print(cnf_matrix[0, 0], "=TP ", cnf_matrix[1, 0], "=FN ", cnf_matrix[0, 1], "=FP ",
          cnf_matrix[1, 1], "=TN")
    print(cnf_matrix)
    return cnf_matrix


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




def F1_macro_average(labels, predictions):
    n_sample = labels.shape[0]
    n_class = np.max(labels + 1)

    P = R = 0
    for i in range(n_class):
        array_labels = np.zeros(n_sample)
        array_predictions = np.zeros(n_sample)

        for j in range(n_sample):
            if labels[j] == i:
                array_labels[j] = 1
            if predictions[j] == i:
                array_predictions[j] = 1

        matrix = confusion_matrix(array_labels, array_predictions)
        TP = matrix[0, 0]
        FP = matrix[0, 1]
        FN = matrix[1, 0]

        P += TP / (TP + FP)  # precision
        R += TP / (TP + FN)  # recall

    P /= n_class
    R /= n_class
    F1 = (2 * P * R) / (P + R)
    return F1




def A_macro_average(labels, predictions):
    n_sample = labels.shape[0]
    n_class = np.max(labels + 1)

    TP = FN = FP = TN= 0
    for i in range(n_class):
        array_labels = np.zeros(n_sample)
        array_predictions = np.zeros(n_sample)

        for j in range(n_sample):
            if labels[j] == i:
                array_labels[j] = 1
            if predictions[j] == i:
                array_predictions[j] = 1

        matrix = confusion_matrix(array_labels, array_predictions)
        TP += matrix[0, 0]
        TN += matrix[1, 1]
        FP += matrix[0, 1]
        FN += matrix[1, 0]

    TP /= n_class
    TN /= n_class
    FP /= n_class
    FN /= n_class

    A = (TP + TN) / (TP + TN + FP + FN)
    return A


def F1_micro_average(labels, predictions):
    matrix = confusion_matrix_multiclass(labels, predictions)
    TP = matrix[0, 0]
    FN = matrix[1, 0]
    FP = matrix[0, 1]

    P = TP / (TP + FP)  # precision
    R = TP / (TP + FN)  # recall

    F1 = (2 * P * R) / (P + R)  # f1 score
    return F1


def A_micro_average(labels, predictions):
    matrix = confusion_matrix_multiclass(labels, predictions)
    TP = matrix[0, 0]
    FN = matrix[1, 0]
    FP = matrix[0, 1]
    TN = matrix[1, 1]

    A = (TP + TN) / (TP + TN + FP + FN)
    return A

