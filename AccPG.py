import numpy as np
from datetime import datetime
from utilities import *
from scipy import sparse

class AccPG:
    def __init__(self, iter, mu, eta, l1Sparsity):
        self.iter = iter
        self.eta = eta
        self.mu = mu
        self.l1Sparsity = l1Sparsity
        

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
        losses = []
        times = []
        for i in range(0, self.iter):
            start = datetime.now()
            gamma = theta * theta * L
            theta = (m - gamma + np.sqrt((gamma - m) * (gamma - m) + 4 * L * gamma)) / (2 * L)
            
            y_s = x_s + theta * gamma / (gamma + m * theta) * (v_s - x_s)
            grad = primal_grad_smooth_hinge_loss_reg(X, sparse.csr_matrix.transpose(X), label, y_s, self.mu / N)
            x_sp = y_s - self.eta * grad
            x_sp = l1_projection(x_sp, self.l1Sparsity)
            v_s = (1 - 1 / theta) * x_s + (1 / theta) * x_sp
            x_s = x_sp
            loss = smooth_hinge_loss_reg(X, label, x_s, self.mu / N)
            end = datetime.now()
            losses.append(loss)
            times.append((end - start).total_seconds())
            print(i, "-iter loss: ", loss)
        print("Prediction Accuracy:", prediction_accuracy(X, label, x_s))
        return losses, times

