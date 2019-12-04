import numpy as np
from scipy import sparse
from utilities import *
from DataLoader import *


class PDBFW:
    def __init__(self):
        self.mu = 10
        self.delta = 20
        self.l0Sparsity = 500
        self.l1Sparsity = 0.5
        self.dualSparsity = 44
        self.L = 1
        self.iter = 100
        self.eta = 0.5
    
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
        
        w_i = np.zeros(N)
        z_i = np.matmul(np.transpose(X), y_i)
        delta_y = np.zeros(N)
        X = sparse.csr_matrix(X)

        
        for current_Iter in range(0, self.iter):
            ########################## Primal ###################
            grad_x_L = z_i + self.mu*x_i
            update_x = x_i - 1/(self.mu * self.L) * grad_x_L
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
                new_y_j_pos = y_i[j] - self.delta + self.delta * w_i[j] / (self.delta + 1)
                new_y_j_neg = y_i[j] + self.delta + self.delta * w_i[j] / (self.delta + 1)
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
            z_i = z_i + np.transpose((sparse.csr_matrix.transpose(X) * sparse.csr_matrix(delta_y.reshape(delta_y.shape[0], 1))).toarray()).reshape(X.shape[1])
        ############################# END ##################
            loss = smooth_hinge_loss_reg(X, label, x_i, self.mu / N)
            print(loss)
        print("prediction_accuracy: " + str(prediction_accuracy(X, label, x_i)) + '\n')
        
X, y = libsvm_load('duke')

ones = np.ones((X.shape[1], 1))
rowsum = (X*ones).reshape((X.shape[0]))
rowsum = np.diag(rowsum)
inv_row = np.linalg.inv(rowsum)
normalized = inv_row * X

pdbfw = PDBFW()
pdbfw.opt(normalized, y)