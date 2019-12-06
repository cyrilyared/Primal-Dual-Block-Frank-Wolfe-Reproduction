from DataLoader import *
from params import *
from AccPG import *
from FW import *
from PDBFW import *
from SCGS import *
from STORC import *
from SVRF import *
from SVRG import *

import numpy as np
import sys

def normalize(X):
    ones = np.ones((X.shape[1], 1))
    rowsum = (X*ones).reshape((X.shape[0]))
    rowsum = np.diag(rowsum)
    inv_row = np.linalg.inv(rowsum)
    normalized = inv_row * X
    return normalized

def printUsage():
    print("Usage: ./main.py <file> <algorithm>")
    print("Files: duke, rcv, mnist, rna")
    print("Algorithms: AccPG, FW, PDBFW, SCGS, STORC, SVRF, SVRG")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        printUsage()
        exit(0)
    
    if sys.argv[1] == "duke":
        file = "duke"
    elif sys.argv[1] == "rcv":
        file = "rcv1_train.binary"
    elif sys.argv[1] == "mnist":
        file = "mnist.09.bin"
    elif sys.argv[1] == "rna":
        file = "mnist.09.bin"
    elif sys.argv[1] == "rna":
        file = "cod-rna"
    else:
        printUsage()
        exit(0)
    
    X, y = libsvm_load(file)
    
    if sys.argv[1] == "duke":
        X = normalize(X)
        param = params.get("duke")
    elif sys.argv[1] == "rcv":
        param = params.get("rcv")
    elif sys.argv[1] == "mnist":
        param = params.get("mnist")
    elif sys.argv[1] == "rna":
        param = params.get("rna")

    if (sys.argv[2] == "AccPG"):
        algo = AccPG(param.get("AccPG_iter"), param.get("mu"), param.get("AccPG_eta"), param.get("l1Sparsity"))
    elif (sys.argv[2] == "FW"):
        algo = FW(param.get("FW_iter"), param.get("mu"), param.get("FW_eta"), param.get("l1Sparsity"))
    elif (sys.argv[2] == "PDBFW"):
        algo = PDBFW(param.get("PDBFW_iter"), param.get("mu"), param.get("PDBFW_eta"), param.get("l1Sparsity"), param.get("PDBFW_l0Sparsity"), param.get("PDBFW_dualSparsity"), param.get("PDBFW_delta"), param.get("PDBFW_L"))
    elif (sys.argv[2] == "SCGS"):
        algo = SCGS(param.get("SCGS_iter"), param.get("mu"), param.get("SCGS_L"), param.get("l1Sparsity"))
    elif (sys.argv[2] == "STORC"):
        algo = STORC(param.get("STORC_iter"), param.get("mu"), param.get("STORC_L"), param.get("l1Sparsity"), param.get("STORC_small_data"))
    elif (sys.argv[2] == "SVRF"):
        algo = SVRF(param.get("SVRF_iter"), param.get("mu"), param.get("SVRF_eta"), param.get("l1Sparsity"))
    elif (sys.argv[2] == "SVRG"):
        algo = SVRG(param.get("SVRG_iter"), param.get("mu"), param.get("SVRG_eta"), param.get("l1Sparsity"))
    else:
        printUsage()
        exit(0)

    losses, times = algo.opt(X, y)
    f = open('./result/' + sys.argv[2], "w")
    f.write(', '.join(map(str, losses)))
    f.write('\n')
    f.write(', '.join(map(str, times)))
    print("Total time (s):", sum(times))
    

    
