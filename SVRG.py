from utilities import *
from DataLoader import *
from scipy import sparse

import numpy as np

class SVRG:
    def __init__(self):
        self.iter = 5
        self.mu = 10
        self.eta = 5
        self.l1Sparsity = 0.5

    def opt(self, X, labels):
        samples = np.shape(X)[0]
        dimensions = np.shape(X)[1]
        indices = np.arange(samples)
        vector_w = np.zeros(dimensions)
        X = sparse.csr_matrix(X)
        for i in range(0, self.iter):
            time_sum = 0
            Nt = min(100, samples)
            vector_gw = primal_grad_smooth_hinge_loss_reg(X, sparse.csr_matrix.transpose(X), labels, vector_w, self.mu / samples)
            vector_x = np.copy(vector_w)
            for j in range(0, Nt):
                m_k = min(100, (int)(samples/10))
                np.random.shuffle(indices)
                grad = primal_grad_smooth_hinge_loss_reg_k(X, labels, vector_x, self.mu / samples, m_k, indices)
                grad_w = primal_grad_smooth_hinge_loss_reg_k(X, labels, vector_w, self.mu / samples, m_k, indices)
                grad = grad - (grad_w - vector_gw)
                vector_x = l1_projection(vector_x - self.eta * grad, self.l1Sparsity)
            vector_w = np.copy(vector_x)
            loss = smooth_hinge_loss_reg(X, labels, vector_x, self.mu / samples)
            print(i, "-th iter loss:", loss)

X, y = libsvm_load('duke')

ones = np.ones((X.shape[1], 1))
rowsum = (X*ones).reshape((X.shape[0]))
rowsum = np.diag(rowsum)
inv_row = np.linalg.inv(rowsum)
normalized = inv_row * X

svrg = SVRG()
svrg.opt(normalized, y)