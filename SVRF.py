from utilities import *
from scipy import sparse
import numpy as np


class SVRF:
    def __init__(self, iter, mu, eta, l1Sparsity):
        self.iter = iter
        self.mu = mu
        self.eta = eta
        self.l1Sparsity = l1Sparsity

    def opt(self, X, label):
        N, D = X.shape  # Number of samples and Number of dimensions
        w_s = np.zeros(D)
        indices = np.arange(N)
        X = sparse.csr_matrix(X)

        for current_iter in range(0, self.iter):
            base = N / 100
            Nt = base - 2
            Gw = primal_grad_smooth_hinge_loss_reg(X, sparse.csr_matrix.transpose(X), label, w_s, self.mu / N)
            x_s = w_s
            for i in range(0, int(Nt)):
                eta = 10.0 / self.l1Sparsity / (i + 3)
                m_k = 100
                np.random.shuffle(indices)
                grad = primal_grad_smooth_hinge_loss_reg_k(X, label, x_s, self.mu / N, m_k, indices)
                gradw = primal_grad_smooth_hinge_loss_reg_k(X, label, w_s, self.mu / N, m_k, indices)
                grad = grad - (gradw - Gw)
                pos, value = 0, 0
                for j in range(0, D):
                    if grad[j] > abs(value) or grad[j] < -abs(value):
                        pos = j
                        value = grad[j]
                x_s = (1 - eta) * x_s
                if value > 0:
                    x_s[pos] -= eta * self.l1Sparsity
                else:
                    x_s[pos] += eta * self.l1Sparsity
                if i % 100 == 0:
                    loss = smooth_hinge_loss_reg(X, label, x_s, self.mu / N)
                    print(current_iter, "iter", i, "th inner loop loss:", loss)
            w_s = x_s
            loss = smooth_hinge_loss_reg(X, label, x_s, self.mu / N)
            print(current_iter, "iter loss:'", loss)
        print("prediction_accuracy:", prediction_accuracy(X, label, w_s))