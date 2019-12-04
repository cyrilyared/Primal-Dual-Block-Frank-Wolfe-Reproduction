import numpy as np
from utilities import *
from DataLoader import *
from scipy import sparse

class AccPG:
    def __init__(self):
        self.iter = 500
        self.eta = 100
        self.mu = 10
        self.l1Spar = 300
        

    def opt(self, X, label):
        N = X.shape[0] # number of samples
        D = X.shape[1]  # number of dimensions
        A = X
        x_s = np.zeros((D))
        y_s = x_s
        v_s = x_s
        L = 1 / self.eta
        m = self.mu / N
        theta = 0
        X = sparse.csr_matrix(X)

        for i in range(0, self.iter):
            gamma = theta * theta * L
            theta = (m - gamma + np.sqrt((gamma - m) * (gamma - m) + 4 * L * gamma)) / (2 * L)
            
            y_s = x_s + theta * gamma / (gamma + m * theta) * (v_s - x_s)
            grad = primal_grad_smooth_hinge_loss_reg(X, sparse.csr_matrix.transpose(X), label, y_s, self.mu / N)
            x_sp = y_s - self.eta * grad
            x_sp = l1_projection(x_sp, self.l1Spar)
            v_s = (1 - 1 / theta) * x_s + (1 / theta) * x_sp
            x_s = x_sp
            loss = smooth_hinge_loss_reg(X, label, x_s, self.mu / N)
            print(loss)


X, y = libsvm_load('rcv1_train.binary')

"""
ones = np.ones((X.shape[1], 1))
rowsum = (X*ones).reshape((X.shape[0]))
rowsum = np.diag(rowsum)
inv_row = np.linalg.inv(rowsum)
normalized = inv_row * X
"""
accpg = AccPG()
accpg.opt(X, y)

