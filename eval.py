#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Classifier
"""

# import arrow
import utils
import torch 
import numpy as np
import robust_classifier as rc
from sklearn.neighbors import KNeighborsClassifier

def get_acc_for_mask(pred, data, test_mask):
    # pred = log_softmax_layer(model(data)).argmax(dim=1)
    correct = (pred[test_mask] == data.y[test_mask]).sum()
    acc = int(correct) / int(test_mask.sum())
    return acc
# TEST METHODS

def knn_regressor(H_test, H_train, p_hat_train, K=5):
    """
    k-Nearest Neighbor Regressor
    Given the train embedding and its corresponding optimal marginal distribution for each class,
    the function produce the prediction of p_hat for testing dataset given the test embedding based
    on the k-Nearest Neighbor rule.
    input
    - H_test:      [n_test_sample, n_feature]
    - H_train:     [n_train_sample, n_feature]
    - p_hat_train: [n_class, n_train_sample]
    output
    - p_hat_test:  [n_class, n_test_sample]
    """
    # find the indices of k-nearest neighbor in trainset
    dist   = utils.pairwise_dist(H_test, H_train)
    dist  *= -1
    _, knb = torch.topk(dist, K, dim=1)        # [n_test_sample, K]
    # print("knb shape", knb.shape)
    # calculate the class marginal probability (p_hat) for each test sample
    p_hat_test = torch.stack(
        [ p_hat_train[:, neighbors].mean(dim=1) 
            for neighbors in knb ], dim=0).t() # [n_class, n_test_sample]
    # print("phat_test shape", p_hat_test.shape)
    return p_hat_test


def simple_knn_regressor(H_test, H_train, Y_train, K=5):
    """
    k-Nearest Neighbor Regressor
    Given the train embedding and its corresponding optimal marginal distribution for each class,
    the function produce the prediction of p_hat for testing dataset given the test embedding based
    on the k-Nearest Neighbor rule.
    input
    - H_test:      [n_test_sample, n_feature]
    - H_train:     [n_train_sample, n_feature]
    - p_hat_train: [n_class, n_train_sample]
    output
    - p_hat_test:  [n_class, n_test_sample]
    """
    # find the indices of k-nearest neighbor in trainset
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(H_train, Y_train)
    p_pred = knn.predict_proba(H_test)

    return p_pred

def kernel_smoother(H_test, H_train, p_hat_train, h=1e-1):
    """
    kernel smoothing test
    Given the train embedding and its corresponding optimal marginal distribution for each class,
    the function produce the prediction of p_hat for testing dataset given the test embedding based
    on the kernel smoothing rule with the bandwidth h.
    input
    - H_test:      [n_test_sample, n_feature]
    - H_train:     [n_train_sample, n_feature]
    - p_hat_train: [n_class, n_train_sample]
    output
    - p_hat_test:  [n_class, n_test_sample]
    """
    n_test_sample, n_feature = H_test.shape[0], H_test.shape[1]
    n_class, n_train_sample  = p_hat_train.shape[0], p_hat_train.shape[1]
    # calculate the pairwise distance between training sample and testing sample
    dist = utils.pairwise_dist(H_train, H_test)   # [n_train_sample, n_test_sample]
    # apply gaussian kernel
    G = 1 / ((np.sqrt(2*np.pi) * h) ** n_feature) * \
        torch.exp(- dist ** 2 / (2 * h ** 2))     # [n_train_sample, n_test_sample]
    G = G.unsqueeze(0).repeat([n_class, 1, 1])    # [n_class, n_train_sample, n_test_sample]
    p_hat_ext  = p_hat_train.unsqueeze(2).\
        repeat([1, 1, n_test_sample])             # [n_class, n_train_sample, n_test_sample]
    p_hat_test = (G * p_hat_ext).mean(dim=1)      # [n_class, n_test_sample]
    return p_hat_test


# TODO
# build a one layer NN mapes the phat to softmax of the p value

def softmax_regressor(H_test, H_train, p_hat_train, K=5):
    """
    k-Nearest Neighbor Regressor
    Given the train embedding and its corresponding optimal marginal distribution for each class,
    the function produce the prediction of p_hat for testing dataset given the test embedding based
    on the k-Nearest Neighbor rule.
    input
    - H_test:      [n_test_sample, n_feature]
    - H_train:     [n_train_sample, n_feature]
    - p_hat_train: [n_class, n_train_sample]
    output
    - p_hat_test:  [n_class, n_test_sample]
    """
    
    m = torch.nn.Softmax(dim=1)
    p_hat_test = m(H_test.t())
    # print("phat_test shape", p_hat_test.shape)
    return p_hat_test


def test(model, data, K=5, h=1e-1, train_test_val = "test", cls_model = None, s_train_mask = None, s_test_mask = None):
    """testing procedure"""

    # given hidden embedding, evaluate corresponding p_hat 
    # using the output of the robust classifier layer
    def evaluate_p_hat(H, Q, theta):
        # print("H, ", H)
        # print("Q, ", Q)

        n_class, n_sample, n_feature = theta.shape[1], H.shape[1], H.shape[2]
        rbstclf = rc.RobustClassifierLayer(n_class, n_sample, n_feature)
        return rbstclf(H, Q, theta).data

    # fetch data from trainset and testset
    # X_train = trainloader.X.float().unsqueeze(1)          # [n_train_sample, 1, n_pixel, n_pixel] 
    # Y_train = trainloader.Y.float().unsqueeze(0)          # [1, n_train_sample]
    # X_test  = testloader.X.float().unsqueeze(1)           # [n_test_sample, 1, n_pixel, n_pixel] 
    # Y_test  = testloader.Y.float()                        # [n_test_sample]
    
    # --- get H (embeddings) and p_hat for trainset and testset --- #
    if train_test_val == "test":
        test_mask = data.test_mask
    elif train_test_val == "train":
        test_mask = data.train_mask
    else:
        test_mask = data.val_mask
    
    # access all the training labels for complete evaluation
    if s_train_mask is not None and s_test_mask is not None:
        # Y_train = data.y[s_train_mask]
        # sorted_index_tr = torch.argsort(Y_train).squeeze()
        # Y_train = torch.index_select(Y_train, 0, sorted_index_tr)
        # # print("Y_train", Y_train.shape)

        # Y_test = data.y[s_test_mask]
        # sorted_index_test = torch.argsort(Y_test).squeeze()
        # Y_test = torch.index_select(Y_test, 0, sorted_index_test)
        # # print("Y_test", Y_test)

        train_mask = s_train_mask
        test_mask = s_test_mask

    else:
        # Y_train = data.y[data.train_mask]
        # sorted_index_tr = torch.argsort(Y_train).squeeze()
        # Y_train = torch.index_select(Y_train, 0, sorted_index_tr)
        # # print("Y_train", Y_train.shape)

        # Y_test = data.y[test_mask]
        # sorted_index_test = torch.argsort(Y_test).squeeze()
        # Y_test = torch.index_select(Y_test, 0, sorted_index_test)
        # # print("Y_test", Y_test)
        
        train_mask = data.train_mask

        if train_test_val == "test":
            test_mask = data.test_mask
        elif train_test_val == "train":
            test_mask = data.train_mask
        else:
            test_mask = data.val_mask


    Y_train = data.y[train_mask]
    sorted_index_tr = torch.argsort(Y_train).squeeze()
    Y_train = torch.index_select(Y_train, 0, sorted_index_tr)
    # print("Y_train", Y_train.shape)

    Y_test = data.y[test_mask]
    sorted_index_test = torch.argsort(Y_test).squeeze()
    Y_test = torch.index_select(Y_test, 0, sorted_index_test)
    # print("Y_test", Y_test)

    model.eval()
    
    with torch.no_grad():
        # all training labels are used to generate the emprical dist.
        Q       = utils.sortedY2Q(Y_train.reshape((1, -1)))   
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Q       = Q.to(device) 
        # Q       = Q.double()          
        # Q       = sortedY2Q(Y_train)                                # [1, n_class, n_sample]
        # print("shape", model.data2vec(data)[data.train_mask].shape)
        model.data2vec.eval()
        H_train = torch.index_select(model.data2vec(data)[train_mask], 0, sorted_index_tr)              # [n_train_sample, n_feature]
        H_test  = torch.index_select(model.data2vec(data)[test_mask], 0, sorted_index_test)             # [n_test_sample, n_feature]
        theta   = model.theta.data.unsqueeze(0)                       # [1, n_class]
        # print(H_train.dtype)
        # print(Q.dtype)
        # Q = Q.float()
        # print(Q.dtype)
        # print(theta.dtype)
        # print(type(H_train.unsqueeze(0)[0]))
        # print(type(Q[0]))
        # print(type(theta[0]))
        p_hat   = evaluate_p_hat(
            H_train.unsqueeze(0), Q, theta).squeeze(0)                # [n_class, n_train_sample]
    
    # print("p_hat", p_hat.shape)
    # --- perform testset pred --- #
    # estimate the probs
    p_hat_knn    = knn_regressor(H_test, H_train, p_hat, K)
    p_hat_kernel = kernel_smoother(H_test, H_train, p_hat, h)   
    p_hat_sft = softmax_regressor(H_test, H_train, p_hat, h)  # [n_class, n_grid * n_grid]
    p_hat_simple_knn = simple_knn_regressor(H_test, H_train, Y_train, K)  # [n_class, n_grid * n_grid]

    # calculate accuracy
    knn_pred            = p_hat_knn.argmax(dim=0)
    knn_n_correct       = knn_pred.eq(Y_test).sum().item()
    knn_accuracy_test   = knn_n_correct / len(Y_test)
    # print("knn_accuracy_test", knn_accuracy_test)

    if cls_model is None:
        # softmax_pred            = p_hat_sft.argmax(dim=0)
        softmax_pred            = H_test.argmax(dim=1)

        # pretrained_gcn.eval()
        # pred = cls(pretrained_gcn(data)).argmax(dim=1)
        # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        # acc = int(correct) / int(data.test_mask.sum())
    else:
        p_hat_sft = cls_model(H_test)
        softmax_pred            = p_hat_sft.argmax(dim=1)
        
    softmax_n_correct       = softmax_pred.eq(Y_test).sum().item()
    softmax_accuracy_test   = softmax_n_correct / len(Y_test)
    # print("softmax pred", softmax_accuracy_test)

    
    # print("Y_test", Y_test.shape)
    simple_knn_pred = torch.tensor(p_hat_simple_knn.argmax(1))
    # print("simple_knn_pred", simple_knn_pred.shape)
    simple_knn_n_correct       = simple_knn_pred.eq(Y_test).sum().item()
    simple_knn_accuracy_test   = simple_knn_n_correct / len(Y_test)
    # print("softmax pred", softmax_accuracy_test)

    kernel_pred         = p_hat_kernel.argmax(dim=0)
    kernel_n_correct    = kernel_pred.eq(Y_test).sum().item()
    kernel_accuracy_test= kernel_n_correct / len(Y_test)
    # print("kernel_accuracy_test pred", kernel_accuracy_test)

    return knn_accuracy_test, kernel_accuracy_test, softmax_accuracy_test, simple_knn_accuracy_test # , knn_accuracy_train



def get_hidden_state(model, data, K=5, h=1e-1, train_test_val = "test", cls_model = None, s_train_mask = None, s_test_mask = None):
    """testing procedure"""

    # given hidden embedding, evaluate corresponding p_hat 
    # using the output of the robust classifier layer
    def evaluate_p_hat(H, Q, theta):
        # print("H, ", H)
        # print("Q, ", Q)

        n_class, n_sample, n_feature = theta.shape[1], H.shape[1], H.shape[2]
        rbstclf = rc.RobustClassifierLayer(n_class, n_sample, n_feature)
        return rbstclf(H, Q, theta).data

    # --- get H (embeddings) and p_hat for trainset and testset --- #
    if train_test_val == "test":
        test_mask = data.test_mask
    elif train_test_val == "train":
        test_mask = data.train_mask
    else:
        test_mask = data.val_mask
    
    # access all the training labels for complete evaluation
    if s_train_mask is not None and s_test_mask is not None:


        train_mask = s_train_mask
        test_mask = s_test_mask

    else:
     
        train_mask = data.train_mask

        if train_test_val == "test":
            test_mask = data.test_mask
        elif train_test_val == "train":
            test_mask = data.train_mask
        else:
            test_mask = data.val_mask


    Y_train = data.y[train_mask]
    sorted_index_tr = torch.argsort(Y_train).squeeze()
    Y_train = torch.index_select(Y_train, 0, sorted_index_tr)
    # print("Y_train", Y_train.shape)

    Y_test = data.y[test_mask]
    sorted_index_test = torch.argsort(Y_test).squeeze()
    Y_test = torch.index_select(Y_test, 0, sorted_index_test)
    # print("Y_test", Y_test)

    model.eval()
    
    with torch.no_grad():
        # all training labels are used to generate the emprical dist.
        Q       = utils.sortedY2Q(Y_train.reshape((1, -1)))   
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Q       = Q.to(device) 
        # Q       = Q.double()          
        # Q       = sortedY2Q(Y_train)                                # [1, n_class, n_sample]
        # print("shape", model.data2vec(data)[data.train_mask].shape)
        model.data2vec.eval()
        H_train = torch.index_select(model.data2vec(data)[train_mask], 0, sorted_index_tr)              # [n_train_sample, n_feature]
        H_test  = torch.index_select(model.data2vec(data)[test_mask], 0, sorted_index_test)             # [n_test_sample, n_feature]
        theta   = model.theta.data.unsqueeze(0)                       # [1, n_class]
        # print(H_train.dtype)
   
        p_hat   = evaluate_p_hat(
            H_train.unsqueeze(0), Q, theta).squeeze(0)                # [n_class, n_train_sample]
    
    return H_test, Y_test

def test_cls_model(cls_model, H_test, Y_test):

    p_hat_sft = cls_model(H_test)
    softmax_pred            = p_hat_sft.argmax(dim=1)
        
    softmax_n_correct       = softmax_pred.eq(Y_test).sum().item()
    softmax_accuracy_test   = softmax_n_correct / len(Y_test)
    # print("softmax pred", softmax_accuracy_test)

    
    return softmax_accuracy_test