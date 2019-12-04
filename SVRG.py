from utilities import *
from datetime import datetime
from DataLoader import *
from scipy import sparse

import numpy as np

class SVRG:
    def __init__(self, iter, mu, eta, l1Sparsity):
        self.iter = iter
        self.mu = mu
        self.eta = eta
        self.l1Sparsity = l1Sparsity

    def opt(self, X, labels):
        samples = np.shape(X)[0]
        dimensions = np.shape(X)[1]
        indices = np.arange(samples)
        vector_w = np.zeros(dimensions)
        X = sparse.csr_matrix(X)
        losses = []
        times = []
        for i in range(0, self.iter):
            start = datetime.now()
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
            end = datetime.now()
            losses.append(loss)
            times.append((end-start).microseconds / 1000000)
            print(i, "-th iter loss:", loss)
        print("Prediction Accuracy:", prediction_accuracy(X, label, vector_x))
        return losses, times