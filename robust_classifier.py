#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Classifier

Note:
In my conda environment, cvxpy can only be installed via pip. If you need to install cvxpy in conda, please refer to the 
following link:
https://stackoverflow.com/questions/41060382/using-pip-to-install-packages-to-anaconda-environment

Dependencies:
- Python      3.7.6
- NumPy       1.18.1
- cvxpy       1.1.0a3
- PyTorch     1.4.0
- cvxpylayers 0.1.2
"""

import torch 
import cvxpy as cp
import numpy as np
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer


class RobustClassifierLayer(torch.nn.Module):
    """
    A Robust Classifier Layer via CvxpyLayer
    """

    def __init__(self, n_class, n_sample, n_feature):
        """
        Initializes the RobustClassifierLayer class.

        Args:
        - n_class: number of classes
        - n_sample: total number of samples in a single batch (including all classes)
        - n_feature: number of features in each sample
        """
        super(RobustClassifierLayer, self).__init__()
        self.n_class = n_class
        self.n_sample = n_sample
        self.n_feature = n_feature
        self.cvxpylayer = self._cvxpylayer(n_class, n_sample)

    def forward(self, X_tch, Q_tch, theta_tch):
        """
        A customized forward function for the RobustClassifierLayer class.

        Args:
        - X_tch: a single batch of the input data
        - Q_tch: the empirical distribution obtained from the labels of this batch
        - theta_tch: a tensor containing the parameters for the model

        Returns:
        - p_hat: the output of the model
        """
        C_tch = self._wasserstein_distance(X_tch)
        gamma_hat = self.cvxpylayer(theta_tch, Q_tch, C_tch)
        gamma_hat = torch.stack(gamma_hat, dim=1)
        p_hat = gamma_hat.sum(dim=2)
        return p_hat

    @staticmethod
    def _wasserstein_distance(X):
        """
        Calculates the wasserstein distance for the input data.

        Args:
        - X: a single batch of the input data

        Returns:
        - C_tch: the wasserstein distance for the input data
        """
        C_tch = []
        for x in X.split(split_size=1):
            x = torch.squeeze(x, dim=0)
            x_norm = (x**2).sum(dim=1).view(-1, 1)
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
            dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
            dist = dist - torch.diag(dist)
            dist = torch.clamp(dist, min=0.0, max=np.inf)
            C_tch.append(dist)
        C_tch = torch.stack(C_tch, dim=0)
        return C_tch


    @staticmethod
    def _cvxpylayer(n_class, n_sample):
        """
        construct a cvxpylayer that solves a robust classification problem
        see reference below for the binary case: 
        http://papers.nips.cc/paper/8015-robust-hypothesis-testing-using-wasserstein-uncertainty-sets
        """
        # NOTE: 
        # cvxpy currently doesn't support N-dim variables, see discussion and solution below:
        # * how to build N-dim variables?
        #   https://github.com/cvxgrp/cvxpy/issues/198
        # * how to stack variables?
        #   https://stackoverflow.com/questions/45212926/how-to-stack-variables-together-in-cvxpy 
        
        # Variables   
        # - gamma_k: the joint probability distribution on Omega^2 with marginal distribution Q_k and p_k
        gamma = [ cp.Variable((n_sample, n_sample)) for k in range(n_class) ]
        # - p_k: the marginal distribution of class k [n_class, n_sample]
        p     = [ cp.sum(gamma[k], axis=0) for k in range(n_class) ] 
        p     = cp.vstack(p) 

        # Parameters (indirectly from input data)
        # - theta: the threshold of wasserstein distance for each class
        theta = cp.Parameter(n_class)
        # - Q: the empirical distribution of class k obtained from the input label
        Q     = cp.Parameter((n_class, n_sample))
        # - C: the pairwise distance between data points
        C     = cp.Parameter((n_sample, n_sample))

        # Constraints
        cons = [ g >= 0. for g in gamma ]
        for k in range(n_class):
            cons += [cp.sum(cp.multiply(gamma[k], C)) <= theta[k]]
            for l in range(n_sample):
                cons += [cp.sum(gamma[k], axis=1)[l] == Q[k, l]]

        # Problem setup
        # tv loss
        obj   = cp.Maximize(cp.sum(cp.min(p, axis=0)))
        # obj   = cp.Minimize(cp.sum(cp.max(p, axis=0)))
        # cross entropy loss
        # obj   = cp.Minimize(cp.sum(- cp.sum(p * cp.log(p), axis=0)))
        prob  = cp.Problem(obj, cons)
        assert prob.is_dpp()

        # return cvxpylayer with shape (n_class [batch_size, n_sample, n_sample])
        # stack operation ('torch.stack(gamma_hat, axis=1)') is needed for the output of this layer
        # to convert the output tensor into a normal shape, i.e., [batch_size, n_class, n_sample, n_sample]
        return CvxpyLayer(prob, parameters=[theta, Q, C], variables=gamma)
    
        gamma_k = cp.Variable((n_class, n_sample, n_sample))
        w_k     = cp.Variable((n_class,))
        
        # Problem constraints
        constraints = []
        
        # 1. gamma_k is a probability distribution
        constraints += [cp.sum(gamma_k[k, :, :]) == 1.0 for k in range(n_class)]
        constraints += [gamma_k[k, :, :] >= 0.0 for k in range(n_class)]
        
        # 2. Wasserstein distance constraints
        # 2.1. w_k >= || mu_0^k - mu_1^k ||_inf 
        for k in range(n_class):
            gamma_k_k   = gamma_k[k, :, :]
            # empirical measures
            mu_0_tch_k  = torch.ones(n_sample, dtype=torch.float32) / n_sample  # uniform measure
            mu_1_tch_k  = torch.squeeze(torch.transpose(Q_tch[:, k, :], 1, 2), dim=1)
            # transform to numpy
            mu_0_k      = mu_0_tch_k.cpu().numpy()
            mu_1_k      = mu_1_tch_k.cpu().numpy()
            # get w_k
            # w_k[k] = np.abs(mu_0_k - mu_1_k).max() # use infinity norm
            w_k[k]      = cp.max(cp.abs(mu_0_k - mu_1_k)) # use infinity norm
            
        # 2.2. gamma_k (i, j) * C_tch(i, j) <= w_k 
        for k in range(n_class):
            gamma_k_k = gamma_k[k, :, :]
            C_tch_k   = C_tch[k, :, :]
            constraints += [cp.sum(gamma_k_k * C_tch_k) <= w_k[k]]
        
        # Solve the problem
        prob = cp.Problem(cp.Minimize(cp.sum(w_k)), constraints)
        # cvxpylayers has an API for wrapping a cvxpy Problem as a PyTorch module. 
        cvxpy_layer = CvxpyLayer(prob, parameters=[theta_tch, Q_tch], variables=[gamma_k])
        
        return cvxpy_layer


# OTHERS


class RobustGraphClassifier(torch.nn.Module):
    """
    A Robust Graph Classifier based on multiple CNNs and a Robust Classifier Layer defined below
    """

    def __init__(self, n_class, n_sample, in_channel = 1, out_channel = 7, max_theta=0.1,
        # in_channel=1, out_channel=7, 
        hidden_size = 16,
        keepprob = 0.9,
        gcn_pre_trained = None,
        theta = None):
        """
        Args:
        - n_class:     number of classes
        - n_sample:    number of sets of samples
        - in_channel:  input channel (input of GCN)
        - out_channel: output channel (output of GCN)
        - max_theta:   threshold for theta_k (empirical distribution)
        - keepprob:    keep probability for dropout layer
                       0.2 in default
        """
        super(RobustGraphClassifier, self).__init__()
        # configurations
        self.n_class   = n_class
        self.max_theta = max_theta
        # self.n_feature = n_feature
        self.in_channel = in_channel


        # # Image to Vec layer
        # self.data2vec  = SimpleImage2Vec(n_feature, 
        #     in_channel, out_channel, n_pixel, kernel_size, stride, keepprob)
        if gcn_pre_trained is not None:
            self.data2vec = gcn_pre_trained
        else:
            self.data2vec = nn.SimpleGCN(in_channel, out_channel, hidden_size = hidden_size, keepprob = keepprob)
            self.data2vec._turn_off_intermediate_value_return()

        # robust classifier layer
        # NOTE: if self.theta is a parameter, then it cannot be reassign with other values, 
        #       since it is one of the attributes defined in the model.
        
        if theta is None:
            self.theta     = torch.nn.Parameter(torch.ones(self.n_class).float() * self.max_theta)
        else:
            self.theta     = torch.nn.Parameter(theta.float())
        self.theta.requires_grad = True
        # self.theta     = torch.ones(self.n_class) * self.max_theta
        self.rbstclf   = RobustClassifierLayer(n_class, n_sample, out_channel)
    
    def forward(self, data, Q, train_flag = True, minibatch_mask = [], sorted_index = []):
        """
        customized forward function.
        input
        - data:  [batch_size, n_sample, in_channel, n_feature]
        - Q:     [batch_size, n_class, n_sample]
        - mask:  []

        output
        - p_hat: [batch_size, n_class, n_sample]
        """
        batch_size  = Q.shape[0] # 1
        n_sample    = Q.shape[2] # 
        # n_feature   = 1477

        # CNN layer
        # NOTE: merge the batch_size dimension and n_sample dimension
        # X = data       # [batch_size*n_sample, in_channel, n_pixel, n_pixel]
        # data = data.view(batch_size* n_sample, self.in_channel, n_feature) # [batch_size, n_sample, n_feature]
        # print("data shape",data.shape)
        Z = self.data2vec(data)                          # [batch_size*n_sample, n_feature]
        # print("original Z size", Z.shape)

        
        if train_flag:
            if len(minibatch_mask) >0:
                Z = Z[minibatch_mask]
                # print("changed z shape to")
                # print(Z.shape)
            else:
                Z = Z[data.train_mask]
        else:
            Z = Z[data.test_mask]
        
        # NOTE: reshape back to batch_size and n_sample
        # print("Z shape", Z.shape)
        Z = Z.view(batch_size, n_sample, Z.shape[-1]) # [batch_size, n_sample, n_feature]
        # print("Z is", Z)
        # print("Q is", Q)
        # print("Z is", Z.shape)
        # print("Q is", Q.shape)
        # robust classifier layer
        # theta = torch.ones(batch_size, self.n_class, requires_grad=True) * self.max_theta

        # Sort Z according to Q
        Z = torch.index_select(Z, 1, sorted_index)
        # print("Z", Z.shape)
        # print("Q", Q.shape)
        theta = self.theta.unsqueeze(0).repeat([batch_size, 1]) # [batch_size, n_class]
        p_hat = self.rbstclf(Z, Q, theta)                       # [batch_size, n_class, n_sample]
        return p_hat

