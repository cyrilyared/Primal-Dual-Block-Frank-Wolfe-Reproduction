import numpy as np
from datetime import datetime
from scipy import sparse
from utilities import *


class PDBFW:
    def __init__(self, iter, mu, eta, l1Sparsity, l0Sparsity, dualSparsity, delta, L):
        self.mu = mu
        self.delta = delta
        self.l0Sparsity = l0Sparsity
        self.l1Sparsity = l1Sparsity
        self.dualSparsity = dualSparsity
        self.L = L
        self.iter = iter
        self.eta = eta
    
    def grad_f_star(self, y, label):
        result = y + label
        for i in range(0, result.shape[0]):
            tmp = y[i] * label[i]
            if tmp >= 0 or tmp <= -1:
                result[i] = 0
        return result

    def opt(self, X, label):
        N = X.shape[0]
        D = X.shape[1]
        x_i = np.zeros(D)
        y_i = np.random.uniform(-1, 1, (N))
        
        X = sparse.csr_matrix(X)
        w_i = np.zeros(N)
        z_i = sparse.csr_matrix.transpose(X).dot(y_i)
        delta_y = np.zeros(N)
        losses = []
        times = []
        
        for current_Iter in range(0, self.iter):
            start = datetime.now()

            ########################## Primal ###################
            grad_x_L = z_i + self.mu*x_i
            update_x = x_i - 1 / (self.mu * self.L) * grad_x_L
            delta_x = l1_l0_projection(update_x, self.l0Sparsity, self.l1Sparsity)
            x_i = (1 - self.eta) * x_i + self.eta * delta_x
            w_i = (1 - self.eta) * w_i + self.eta * (X * (sparse.csr_matrix(delta_x.reshape(delta_x.shape[0], 1)))).toarray().reshape((X.shape[0]))
            ########################## Dual #####################
            coordin_selector = w_i - self.grad_f_star(y_i, label)
            coordin_selector = l0_projection(coordin_selector, self.dualSparsity)
            delta_y = np.zeros(N)
            for j in range(0, N):
                if coordin_selector[j] == 0:
                    continue
                new_y_j_pos = (y_i[j] - self.delta + self.delta * w_i[j]) / (self.delta + 1)
                new_y_j_neg = (y_i[j] + self.delta + self.delta * w_i[j]) / (self.delta + 1)
                if (label[j] == - 1):
                    new_y_j = min(max(new_y_j_neg, 0.0), 1.0)
                elif (label[j] == 1):
                    new_y_j = min(max(new_y_j_pos, -1.0), 0.0)
                else:
                    print("label" + str(j) + " is "  + str(label[j]) + "\n")
                    print ("INPUT ERROR")
                    exit(0)
                delta_y[j] = new_y_j - y_i[j]
                y_i[j] = new_y_j
            z_i = z_i + (sparse.csr_matrix.transpose(X).dot(sparse.csr_matrix(delta_y.reshape(delta_y.shape[0], 1)))).toarray().reshape(X.shape[1])
        ############################# END ##################
            loss = smooth_hinge_loss_reg(X, label, x_i, self.mu / N)
            end = datetime.now()
            losses.append(loss)
            times.append((end-start).total_seconds())
            print(current_Iter, "-iter loss: ", loss)
        print("Prediction Accuracy: " + str(prediction_accuracy(X, label, x_i)))
        return losses, times