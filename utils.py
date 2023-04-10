#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities

Dependencies:
- Python      3.7.6
- NumPy       1.18.1
- PyTorch     1.4.0
- arrow       0.13.1
"""

import torch 
import numpy as np
import torch
from plots import *
import random

def seed_everything(seed=42):
    """
    Seed random number generators for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tvloss(p_hat):
    """TV loss"""
    # p_max, _ = torch.max(p_hat, dim=1) # [batch_size, n_class]
    # return p_max.sum(dim=1).mean()     # scalar
    p_min, _ = torch.min(p_hat, dim=1) # [batch_size, n_sample]
    return p_min.sum(dim=1).mean()     # scalar
    

def celoss(p_hat):
    """cross entropy loss"""
    crossentropy = - p_hat * torch.log(p_hat) # [batch_size, n_sample]
    return crossentropy.sum(dim=1).mean()     # scalar

def pairwise_dist(X, Y):
    """
    calculate pairwise l2 distance between X and Y
    """
    X_norm = (X**2).sum(dim=1).view(-1, 1)            # [n_xsample, 1]
    Y_t    = torch.transpose(Y, 0, 1)                 # [n_feature, n_ysample]
    Y_norm = (Y**2).sum(dim=1).view(1, -1)            # [1, n_ysample]
    dist   = X_norm + Y_norm - 2.0 * torch.mm(X, Y_t) # [n_xsample, n_ysample]
    return dist 

def sortedY2Q(Y):
    """
    transform the sorted input label into the empirical distribution matrix Q, where
        Q_k^l = 1 / n_k, for n_{k-1} \le l \le n_{k+1}
              = 0, otherwise

    input
    - Y: [batch_size, n_sample]
    output
    - Q: [batch_size, n_class, n_sample]
    """
    batch_size, n_sample = Y.shape
    # NOTE:
    # it is necessary to require that the number of data points of each class in a single batch 
    # is no less than 1 here.
    classes = torch.unique(Y)
    n_class = classes.shape[0]
    # N records the number of data points of each class in each batch [batch_size, n_class]
    N = [ torch.unique(y, return_counts=True)[1] for y in Y.split(split_size=1) ]
    N = torch.stack(N, dim=0)
    # construct an empty Q matrix with zero entries
    Q = torch.zeros(batch_size, n_class, n_sample)
    for batch_idx in range(batch_size):
        for class_idx in range(n_class):
            _from = N[batch_idx, :class_idx].sum()
            _to   = N[batch_idx, :class_idx+1].sum()
            n_k   = N[batch_idx, class_idx].float()
            Q[batch_idx, class_idx, _from:_to] = 1 / n_k
    Q.float()
    # print("q dtype", Q.dtype)
    return Q

def evaluate_2Dspace(X_train, X_test, n_grid):
    min_X        = np.concatenate((X_train, X_test), 0).min(axis=0)
    max_X        = np.concatenate((X_train, X_test), 0).max(axis=0)
    min_X, max_X = min_X - (max_X - min_X) * .2, max_X + (max_X - min_X) * .2
    X_space      = [ np.linspace(min_x, max_x, n_grid + 1)[:-1] 
        for min_x, max_x in zip(min_X, max_X) ]           # (n_feature [n_grid])
    X            = [ [x1, x2] for x1 in X_space[0] for x2 in X_space[1] ]
    X            = torch.Tensor(X)                        # [n_grid * n_grid, n_feature]
    return min_X, max_X, X