import numpy as np
from utilities import *
from DataLoader import *
from scipy import sparse

class SCGS:
    def __init__(self):
        self.iter = 50
        self.L = 0.5
        self.mu = 10
        self.l1Spar = 0.5
        

    def opt(self, X, label):
        N = X.shape[0] # number of samples
        D = X.shape[1]  # number of dimensions
        A = X
        z_s = np.zeros((D))
        q_s = z_s
        w_s = z_s
        X = sparse.csr_matrix(X)
        for i in range(0, self.iter):
            gamma = 3.0 / (i + 3)
            z_s = (1 - gamma) * w_s + gamma * q_s
            m_k = min(N, max(100, int(N / 20)))
            indices = np.arange(N)
            np.random.shuffle(indices)
            Gz = primal_grad_smooth_hinge_loss_reg_k(A, label, z_s, self.mu / N, m_k, indices)
            q_s = q_s - (i + 1) / (3 * self.L) * Gz
            q_s = l1_projection(q_s, self.l1Spar)
            w_s = (1-gamma)*w_s + gamma * q_s
            loss = smooth_hinge_loss_reg(X, label, w_s, self.mu / N)
            print(loss)


X, y = libsvm_load('duke')

ones = np.ones((X.shape[1], 1))
rowsum = (X*ones).reshape((X.shape[0]))
rowsum = np.diag(rowsum)
inv_row = np.linalg.inv(rowsum)
normalized = inv_row * X

scgs = SCGS()
scgs.opt(normalized, y)

