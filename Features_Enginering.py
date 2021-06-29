import numpy as np

def normalize_transform(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / (max_value - min_value)
    return to_normalize


def center_transform(to_normalize, column_index, mean_value):
    to_normalize[:, column_index] -= mean_value
    return to_normalize


def sig_transform(to_normalize, column_index, b):
    to_normalize[:, column_index] = 1 / (1 + np.exp(b * to_normalize[:, column_index]))
    return to_normalize


def log_transform(to_normalize, column_index, b):
    to_normalize[:, column_index] = np.log(b + to_normalize[:, column_index])
    return to_normalize


def clip_transform(to_normalize, column_index, b, up_or_down="up"):
    for i in range(to_normalize.shape[0]):
        print(to_normalize[i, column_index])
        if up_or_down == "down" or up_or_down == "up_down":
            to_normalize[i, column_index] = np.sign(pow(to_normalize[i, column_index], 2)) * np.max(
                [abs(to_normalize[i, column_index]), b])
        if up_or_down == "up" or up_or_down == "up_down":
            to_normalize[i, column_index] = np.sign(pow(to_normalize[i, column_index], 2)) * np.min(
                [abs(to_normalize[i, column_index]), b])
    return to_normalize