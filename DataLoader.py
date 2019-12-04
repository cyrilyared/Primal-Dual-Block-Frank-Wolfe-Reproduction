import numpy as np
from sklearn.datasets import load_svmlight_file

def readData(file, delimiter='\t', first_col_lbl=True):
    data = np.genfromtxt(file, delimiter=delimiter)
    if first_col_lbl is True:
        y = data[:, 0:1]
        X = data[:, 1:]
    else:
        y = data[:, -1:]
        X = data[:, 0:-1]
    return X, y

def libsvm_load(file):
    return load_svmlight_file(file)