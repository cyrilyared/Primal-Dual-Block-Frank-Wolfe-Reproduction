# Primal Dual Block Frank Wolfe Baselines Reproduction

* Usage: `main.py <dataset> <algorithm>`
* Algorithms:
    * `AccPG` - Accelerated Projected Gradient
    * `FW` - Frank-Wolfe
    * `SCGS` - Stochastic Conditional Gradient Sliding
    * `STORC` - Stochastic Variance-Reduced Contional Gradient Sliding
    * `SVRG` - Stochastic Variance-Reduced Gradient
    * `PDBFW` - Primal-Dual Block Frank-Wolfe
    * `SVRF` - Stochastic Variance-Reduced Frank-Wolfe
* Datasets:
    * `duke` - Duke Cancer Dataset
    * `rcv` - Reuters Corpus Volume I Dataset
    * `mnist` - Modified MNIST Dataset
    * `rna` - Non-Coding RNA Detection Dataset
* Parameters: All hyperparameters can be edited in `params.py`.

Original Paper: https://arxiv.org/pdf/1906.02436.pdf
Citation: 