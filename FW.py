from utilities import *
from datetime import datetime
from scipy import sparse
import numpy as np

class FW:
    def __init__(self, iter, mu, eta, l1Sparsity):
        self.iter = iter
        self.mu = mu
        self.eta = eta
        self.l1Sparsity = l1Sparsity


    def opt(self, X, label):
        N, D = X.shape  # Number of samples and Number of dimensions
        eigen_vector = np.zeros((D))
        X = sparse.csr_matrix(X)
        losses = []
        times = []
        for current_iter in range(0, self.iter):
            start = datetime.now()

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
            end = datetime.now()
            losses.append(loss)
            times.append((end-start).microseconds / 1000000)
            print(current_iter, "-iter loss: ", loss)
        print("Prediction Accuracy:", prediction_accuracy(X, label, eigen_vector))
        return losses, times