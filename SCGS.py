import numpy as np
from datetime import datetime
from utilities import *
from scipy import sparse

class SCGS:
    def __init__(self, iter, mu, L, l1Sparsity):
        self.iter = iter
        self.L = L
        self.mu = mu
        self.l1Sparsity = l1Sparsity
        

    def opt(self, X, label):
        N = X.shape[0] # number of samples
        D = X.shape[1]  # number of dimensions
        A = X
        z_s = np.zeros((D))
        q_s = z_s
        w_s = z_s
        X = sparse.csr_matrix(X)
        losses = []
        times = []
        for i in range(0, self.iter):
            start = datetime.now()
            gamma = 3.0 / (i + 3)
            z_s = (1 - gamma) * w_s + gamma * q_s
            m_k = min(N, max(100, int(N / 20)))
            indices = np.arange(N)
            np.random.shuffle(indices)
            Gz = primal_grad_smooth_hinge_loss_reg_k(A, label, z_s, self.mu / N, m_k, indices)
            q_s = q_s - (i + 1) / (3 * self.L) * Gz
            q_s = l1_projection(q_s, self.l1Sparsity)
            w_s = (1-gamma)*w_s + gamma * q_s
            loss = smooth_hinge_loss_reg(X, label, w_s, self.mu / N)
            end = datetime.now()
            losses.append(loss)
            times.append((end-start).microseconds / 1000000)
            print(i, "-iter loss: ", loss)
        print("Prediction Accuracy:", prediction_accuracy(X, label, w_s))
        return losses, times

