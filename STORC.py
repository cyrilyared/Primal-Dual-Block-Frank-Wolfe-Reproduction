
from utilities import *
from datetime import datetime
from scipy import sparse
import numpy as np


class STORC:
    def __init__(self, iter, mu, L, l1Sparsity, small_data):
        self.iter = iter
        self.small_data = small_data
        self.L = L
        self.l1Sparsity = l1Sparsity
        self.mu = mu
        
    def opt(self, X, label):
        N, D = X.shape  # Number of samples and Number of dimensions
        indices = np.arange(N)
        x_s = np.zeros(D)
        w_s = x_s
        base = 8
        X = sparse.csr_matrix(X)
        losses = []
        times = []
        for current_iter in range(0, self.iter):
            start = datetime.now()
            Nt = min(min(N / 20, 200), base * 2 - 2)
            if self.small_data:
                Nt = min(min((N / 5), 200), base * 2 - 2)
            base *= 2
            y_s = w_s
            Gy = primal_grad_smooth_hinge_loss_reg(X, sparse.csr_matrix.transpose(X), label, y_s, self.mu / N)
            x_s = y_s
            for k in range(0, (int)(Nt)):
                beta = 3.0 * self.L / (k + 1)
                gamma = 2.0 / (k + 2)
                z_s = (1 - gamma) * y_s + gamma * x_s
                m_k = min(N, 5 * (k + 1))
                np.random.shuffle(indices)
                grad = primal_grad_smooth_hinge_loss_reg_k(X, label, z_s, self.mu / N, m_k, indices)
                grady = primal_grad_smooth_hinge_loss_reg_k(X, label, w_s, self.mu / N, m_k, indices)
                grad = grad - (grady - Gy)
                x_s = l1_projection(-grad / beta + x_s, self.l1Sparsity)
                y_s = (1 - gamma) * y_s + gamma * x_s
            w_s = x_s
            loss = smooth_hinge_loss_reg(X, label, x_s, self.mu / N)
            end = datetime.now()
            losses.append(loss)
            times.append((end-start).total_seconds())
            print(current_iter, "-iter loss: ", loss)
        print("Prediction Accuracy:", prediction_accuracy(X, label, x_s))
        return losses, times
