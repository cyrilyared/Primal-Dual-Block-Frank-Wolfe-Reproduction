from utilities import *
from DataLoader import *
from scipy import sparse
import numpy as np

class FW:
    def __init__(self):
        self.iter = 500
        self.mu = 10
        self.eta = 0.5
        self.l1Sparsity = 0.5


    def opt(self, X, label):
        N, D = X.shape  # Number of samples and Number of dimensions
        eigen_vector = np.zeros((D))
        X = sparse.csr_matrix(X)
        for current_iter in range(0, self.iter):
            # Primal
            eta = self.eta * 2.0 / (current_iter + 3)
            grad = primal_grad_smooth_hinge_loss_reg(X, sparse.csr_matrix.transpose(X), label, eigen_vector, self.mu / N)
            pos = 0
            value = 0
            for i in range(0, D):
                if grad[i] > abs(value) or grad[i] < -abs(value):
                    pos = i
                    value = grad[i]
            eigen_vector *= 1 - eta
            if value > 0:
                eigen_vector[pos] -= eta * self.l1Sparsity
            else:
                eigen_vector[pos] += eta * self.l1Sparsity
            loss = smooth_hinge_loss_reg(X, label, eigen_vector, self.mu / N)
            print(current_iter, "iter loss:", loss)

        print("prediction_accuracy:", prediction_accuracy(X, label, eigen_vector))

X, y = libsvm_load('duke')

ones = np.ones((X.shape[1], 1))
rowsum = (X*ones).reshape((X.shape[0]))
rowsum = np.diag(rowsum)
inv_row = np.linalg.inv(rowsum)
normalized = inv_row * X

fw = FW()
fw.opt(normalized, y)