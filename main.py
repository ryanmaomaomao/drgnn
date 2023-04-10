#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import torchsettings.json
import argparse
import copy
import utils
import dataloader
import numpy as np
import eval
from tqdm.auto import tqdm
import nn
from sklearn.neighbors import KNeighborsClassifier
import robust_classifier as rc
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
import copy
from plots import *
from trainer import *


def drgcn_exp_main():
    print("Running drgcn_exp_main.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    utils.seed_everything(2)

    # model config
    sample_per_class    = 5                             # number of samples per class given
    dataset             = Planetoid(root='/tmp/Cora', name='Cora', num_train_per_class = sample_per_class, split = 'random')
    n_class             = 7                             # number classes given
    classes             = [i for i in range(n_class)]   # listing of classes
    in_feature          = dataset.num_node_features     # the input feature length
    n_feature           = 2                             # the gcn output embedding dim
    batch_class_size    = 5                             # used part of all samples of a class per mini-set
                                                        # should be smaller than sample_per_class
    n_sample            = n_class * batch_class_size    # per batch number of samples used for cvs optimization
    max_theta           = 1e-2                          # maximum wasserstain distance for samples within a class
    lr                  = 0.1                           # lr
    K                   = 14                            # k nearest
    gcn_hidden_dim      = 16
    # batch_size          = 10                            # per batch number of sets
    # n_iter              = 101
    # gap = 20

    # init model
    # model       = rc.RobustGraphClassifier(n_class, n_sample, n_feature, max_theta)

    # init data
    data = dataset[0].to(device)

    # ------------------------------ GCN pretraining ------------------------------ #
    print("# ------------------------------ GNN pretraining ------------------------------ #")

    # pretrained_gcn = SimpleGCN(in_feature = dataset.num_node_features, out_feature = n_feature,
    #                            hidden_size = 16)
    pretrained_gcn = nn.GCN(input_dim = dataset.num_node_features, hidden_dim = gcn_hidden_dim)
    # pretrained_gcn = nn.GAT(num_features = dataset.num_node_features, hidden_dim = 56,
    # num_layers = 2,
    # num_classes = 7,
    # dropout = 0.6, heads = 8)
    NN_cls = nn.NN_classifier(input_dim = 7, output_dim = 7, keepprob = 0.9, linear_hidden_size=512)
    log_softmax_layer = nn.ClassficationLayer()

    train_mask = np.isin(data.y.cpu(), classes) * data.train_mask.cpu().numpy()
    test_mask = np.isin(data.y.cpu(), classes) * data.test_mask.cpu().numpy()
    val_mask = np.isin(data.y.cpu(), classes) * data.val_mask.cpu().numpy()

    # pretrained_gcn = SimpleGCN(in_feature =  , out_feature = 7, return_intermediate_value = False).to(device)
    # data = dataset[0].to(device)
    optimizer = torch.optim.Adam([
                    {'params': pretrained_gcn.parameters()},
                    {'params': NN_cls.parameters()}
                ], lr=0.01, weight_decay=5e-4)

    pretrained_gcn.to(device)
    NN_cls.to(device)
    log_softmax_layer.to(device)

    for epoch in range(1000):
        pretrained_gcn.train()
        NN_cls.train()
        optimizer.zero_grad()
        out = log_softmax_layer(NN_cls(pretrained_gcn(data)))
        # print(pretrained_gcn(data))
        # print(out.shape) # output shape: [2708, 7]
        # print(out[train_mask].shape) # output shape: [35, 7]
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        # print(out[train_mask].shape)
        # print(data.y[train_mask].shape)
        loss.backward()
        optimizer.step()
    
    # testing
    pretrained_gcn.eval()
    pred = log_softmax_layer(NN_cls(pretrained_gcn(data))).argmax(dim=1)
    print(f'Pretrain GNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain GNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')
    knn = KNeighborsClassifier(n_neighbors=K)

    X = pretrained_gcn(data)[train_mask].cpu().detach().numpy()
    y = data.y[train_mask].cpu().detach().numpy()
    knn.fit(X, y)
    pred = torch.tensor(knn.predict(pretrained_gcn(data).cpu().detach().numpy()))
    print(f'Pretrain KNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain KNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')

    # ------------------------------ DR training ------------------------------ #
    print("# ------------------------------ DR training ------------------------------ #")

    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    loss_list = []
    best_val_score = 0
    gcn_acc_list = []

    gap = 20
    batch_size = 20
    num_epochs = 501
    dr_early_stopping = False

    model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = pretrained_gcn).to(device)
    # model = rc.RobustGraphClassifier(n_class, n_sample, in_feature, out_channel = 7, hidden_size = 16).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
    flag_insolvant = False
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            Q = utils.sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            Q = Q.to(device)
            try:
                p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
                            sorted_index = sorted_index.squeeze())
            
                # loss = F.nll_loss(out, Y_train.reshape((1,35)))
                # TV Loss
                loss += utils.tvloss(p_hat)
                # loss  = celoss(p_hat)
            except:
                print("Insolvant error! Skip and stop.")
                flag_insolvant = True
                break
        if flag_insolvant:
            break
        else:
            loss/=batch_size
            loss_list.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            
        # model.eval()
        # pred = model(data, Q).argmax(dim=1)
        # print("pred shape", pred.shape)
        # pred = pred.reshape(35)
        # correct = (pred == data.y[data.train_mask]).sum()
        # acc = int(correct) / int(data.test_mask.sum())


        # print(loss_list)
        # print(np.mean(loss_list))
        if dr_early_stopping:
            if epoch % gap == 0:
                # loss_list.append(loss.item()/batch_size)
                # knn_acc_val, kernel_acc_val = test(model, data, K=K, h=1e-2, train_test_val = "train")
                # kernel_acc_list.append(kernel_acc_val)
                # knn_acc_list.append(knn_acc_val)

                knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(model, data, K=K, h=1e-2, train_test_val = "val",
                                                                    s_train_mask = train_mask, s_test_mask = val_mask)
                # knn_acc_train, kernel_acc_train = test(model, data, K=K, h=1e-2, train_test_val = "train")

                # print("Train loss: {:.6f} \nVal set ({:d} samples):  kNN accuracy: {:.3f}, kernel smoothing accuracy: {:.3f}, Softmax acc: {:.3f} \n".format(loss.item()/batch_size, \
                #                                             sum(data.val_mask), knn_acc_val, kernel_acc_val, gcn_acc_val))
                # print("Train kNN accurary: %.3f\nTest set (%d samples):  kNN accuracy: %.3f, kernel smoothing accuracy: %.3f\n" 
                #       % (knn_accuracy_train, sum(data.test_mask), knn_accuracy, kernel_accuracy))
                
                with tqdm(total=1, disable=True) as pbar:
                    tqdm.write(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, KNN val acc={knn_acc_val:.4f}, Kernel val acc={kernel_acc_val:.4f}, Softmax val acc ={gcn_acc_val:.4f}")
            
                knn_acc_list.append(knn_acc_val)
                kernel_acc_list.append(kernel_acc_val)
                gcn_acc_list.append(gcn_acc_val)


                # knn_train_acc_list.append(knn_acc_train)

                if knn_acc_val > best_val_score:
                    best_val_score = knn_acc_val
                    # model_copy = copy.copy(model) 
                    # model_copy = model
                    # print("model replaced")
                    gcn_model_copy = copy.deepcopy(model.data2vec) 
                    theta_copy = model.theta.clone()

                    # best_test_acc, _, _ = eval.test(model, data, K=K, h=1e-2, train_test_val = "test",
                    #                                                 s_train_mask = train_mask, s_test_mask = test_mask)
    if not dr_early_stopping:
        gcn_model_copy = copy.deepcopy(model.data2vec) 
        theta_copy = model.theta.clone()

    # print(best_test_acc)
    best_model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = gcn_model_copy, theta = theta_copy).to(device)
    best_knn_test_acc, best_kernel_acc_val, best_softmax_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "test",
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    print("best_knn_test_acc", best_knn_test_acc)
    # print(loss_list)
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([knn_acc_list], gap = gap, legend_list = ["knn_acc_list"])

    # ------------------------------ CLS training ------------------------------ #
    print("# ------------------------------ CLS training ------------------------------ #")

    # train nn to fit the dimension
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(NN_cls.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam([
                # {'params': model.parameters()},
                {'params': NN_cls.parameters()}
            ], lr=1e-3, weight_decay=5e-4)
    
    H_val, Y_val = eval.get_hidden_state(best_model, data, K=K, h=1e-2, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)

    best_val_score = 0

    gap = 20
    batch_size = 20
    n_iter = 501

    loss_list = []
    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    gcn_acc_list = []

    for epoch in tqdm(range(n_iter)):
        NN_cls.train()
        # model.eval()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            # print(Y_train.shape)
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            # print(Y_train.shape)
            # Q = sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            # Q = Q.to(device)
            # p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
            #             sorted_index = sorted_index.squeeze())
            # loss = F.nll_loss(out, Y_train.reshape((1,35)))
            # TV Loss
            # loss  = celoss(p_hat)

            # cls_.eval()
            # model.eval()
            
            # with torch.no_grad():
            # all training labels are used to generate the emprical dist.
            # Q       = sortedY2Q(Y_train.reshape((1, n_class * 3)))      
            # Q       = Q.to(device) 
            with torch.no_grad():
                H_train  = torch.index_select(best_model.data2vec(data)[mask], 0, sorted_index)             # [n_test_sample, n_feature]
            # m = torch.nn.Softmax(dim=1)
            # p_hat_test = m(H_train.t())
            # print("phat_test shape", p_hat_test.shape)
            # # return p_hat_test
            # print("p_hat", p_hat.shape)
            # print("H_train t", H_train.t().shape)


            # --- perform testset pred --- #
            # estimate the probs
            # p_hat_sft = softmax_regressor(H_train, H_train, p_hat, h)  # [n_class, n_grid * n_grid]
            p_hat_sft = NN_cls(H_train)
            # softmax_pred            = p_hat_sft.argmax(dim=1)
            # print("softmax_pred", p_hat_sft.shape)
            # print("Y_train", Y_train.shape)

            loss += criterion(p_hat_sft, Y_train)
            
        loss /= batch_size
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item()/batch_size)

        # print(loss_list)
        # print(np.mean(loss_list))
        if epoch % gap == 0:
            # knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)
            gcn_acc_val = eval.test_cls_model(NN_cls, H_val, Y_val)
            # kernel_acc_list.append(kernel_acc_val)
            # knn_acc_list.append(knn_acc_val)
            # knn_train_acc_list = []
        
            gcn_acc_list.append(gcn_acc_val)
            with tqdm(total=1, disable=True) as pbar:
                tqdm.write(f"Epoch {epoch+1}/{n_iter}: Loss={loss:.4f}, gcn_acc_val, Softmax val acc ={gcn_acc_val:.4f}")
                # tqdm.write(f"Epoch {epoch+1}/{n_iter}: Loss={loss:.4f}, KNN val acc={knn_acc_val:.4f}, Kernel val acc={kernel_acc_val:.4f}, Softmax val acc ={gcn_acc_val:.4f}")
        
            if gcn_acc_val > best_val_score:
                best_val_score = gcn_acc_val
                # _, _, best_test_score = eval.test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = test_mask)
                # _, _, best_test_score = test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = cls_)
                cls_model_copy = copy.deepcopy(NN_cls) 
    
    # print("best_test_score", best_test_score)
    _, _, best_softmax_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "test", cls_model = cls_model_copy,
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([gcn_acc_list], gap = gap, legend_list = ["gcn_acc_val"])

    print("best_softmax_acc_val", best_softmax_acc_val)



def plot_main():
    print("Running plot main.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    utils.seed_everything(0)

    # model config
    sample_per_class    = 5                             # number of samples per class given
    dataset             = Planetoid(root='/tmp/Cora', name='Cora', num_train_per_class = sample_per_class, split = 'random')
    n_class             = 3                             # number classes given
    classes             = [i for i in range(n_class)]   # listing of classes
    in_feature          = dataset.num_node_features     # the input feature length
    n_feature           = 2                             # the gcn output embedding dim
    batch_class_size    = 4                             # used part of all samples of a class per mini-set
                                                        # should be smaller than sample_per_class
    n_sample            = n_class * batch_class_size    # per batch number of samples used for cvs optimization
    max_theta           = 1e-2                          # maximum wasserstain distance for samples within a class
    lr                  = 0.1                           # lr
    K                   = 3                            # k nearest
    gcn_hidden_dim      = 16
    h                   = 1
    # batch_size          = 10                            # per batch number of sets
    # n_iter              = 101
    # gap = 20

    # init model
    # model       = rc.RobustGraphClassifier(n_class, n_sample, n_feature, max_theta)

    # init data
    data = dataset[0].to(device)

    # ------------------------------ GCN pretraining ------------------------------ #
    print("# ------------------------------ GNN pretraining ------------------------------ #")

    # pretrained_gcn = SimpleGCN(in_feature = dataset.num_node_features, out_feature = n_feature,
    #                            hidden_size = 16)
    pretrained_gcn = nn.GCN(input_dim = dataset.num_node_features, hidden_dim = gcn_hidden_dim, output_dim=n_feature)
    # pretrained_gcn = nn.GAT(num_features = dataset.num_node_features, hidden_dim = 56,
    # num_layers = 2,
    # num_classes = 7,
    # dropout = 0.6, heads = 8)
    NN_cls = nn.NN_classifier(input_dim = n_feature, output_dim = n_class, keepprob = 0.9, linear_hidden_size=128)
    log_softmax_layer = nn.ClassficationLayer()

    train_mask = np.isin(data.y.cpu(), classes) * data.train_mask.cpu().numpy()
    test_mask = np.isin(data.y.cpu(), classes) * data.test_mask.cpu().numpy()
    val_mask = np.isin(data.y.cpu(), classes) * data.val_mask.cpu().numpy()

    # pretrained_gcn = SimpleGCN(in_feature =  , out_feature = 7, return_intermediate_value = False).to(device)
    # data = dataset[0].to(device)
    optimizer = torch.optim.Adam([
                    {'params': pretrained_gcn.parameters()},
                    {'params': NN_cls.parameters()}
                ], lr=0.01, weight_decay=5e-4)

    pretrained_gcn.to(device)
    NN_cls.to(device)
    log_softmax_layer.to(device)

    for epoch in range(1000):
        pretrained_gcn.train()
        NN_cls.train()
        optimizer.zero_grad()
        out = log_softmax_layer(NN_cls(pretrained_gcn(data)))
        # print(pretrained_gcn(data))
        # print(out.shape) # output shape: [2708, 7]
        # print(out[train_mask].shape) # output shape: [35, 7]
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        # print(out[train_mask].shape)
        # print(data.y[train_mask].shape)
        loss.backward()
        optimizer.step()
    
    # testing
    pretrained_gcn.eval()
    pred = log_softmax_layer(NN_cls(pretrained_gcn(data))).argmax(dim=1)
    print(f'Pretrain GNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain GNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')
    knn = KNeighborsClassifier(n_neighbors=K)

    X = pretrained_gcn(data)[train_mask].cpu().detach().numpy()
    y = data.y[train_mask].cpu().detach().numpy()
    knn.fit(X, y)
    pred = torch.tensor(knn.predict(pretrained_gcn(data).cpu().detach().numpy()))
    print(f'Pretrain KNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain KNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')

    # ------------------------------ DR training ------------------------------ #
    print("# ------------------------------ DR training ------------------------------ #")

    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    loss_list = []
    best_val_score = 0
    gcn_acc_list = []

    gap = 20
    batch_size = 20
    num_epochs = 201

    model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = pretrained_gcn).to(device)
    # model = rc.RobustGraphClassifier(n_class, n_sample, in_feature, out_channel = 7, hidden_size = 16).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
    flag_insolvant = False
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            Q = utils.sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            Q = Q.to(device)
            try:
                p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
                            sorted_index = sorted_index.squeeze())
            
                # loss = F.nll_loss(out, Y_train.reshape((1,35)))
                # TV Loss
                loss += utils.tvloss(p_hat)
                # loss  = celoss(p_hat)
            except:
                print("Insolvant error! Skip and stop.")
                flag_insolvant = True
                break
        if flag_insolvant:
            break
        else:
            loss/=batch_size
            loss_list.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            
        # model.eval()
        # pred = model(data, Q).argmax(dim=1)
        # print("pred shape", pred.shape)
        # pred = pred.reshape(35)
        # correct = (pred == data.y[data.train_mask]).sum()
        # acc = int(correct) / int(data.test_mask.sum())


        # print(loss_list)
        # print(np.mean(loss_list))
        if epoch % gap == 0:
            # loss_list.append(loss.item()/batch_size)
            # knn_acc_val, kernel_acc_val = test(model, data, K=K, h=1e-2, train_test_val = "train")
            # kernel_acc_list.append(kernel_acc_val)
            # knn_acc_list.append(knn_acc_val)

            knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(model, data, K=K, h=h, train_test_val = "val",
                                                                s_train_mask = train_mask, s_test_mask = val_mask)
            # knn_acc_train, kernel_acc_train = test(model, data, K=K, h=1e-2, train_test_val = "train")

            # print("Train loss: {:.6f} \nVal set ({:d} samples):  kNN accuracy: {:.3f}, kernel smoothing accuracy: {:.3f}, Softmax acc: {:.3f} \n".format(loss.item()/batch_size, \
            #                                             sum(data.val_mask), knn_acc_val, kernel_acc_val, gcn_acc_val))
            # print("Train kNN accurary: %.3f\nTest set (%d samples):  kNN accuracy: %.3f, kernel smoothing accuracy: %.3f\n" 
            #       % (knn_accuracy_train, sum(data.test_mask), knn_accuracy, kernel_accuracy))
            
            with tqdm(total=1, disable=True) as pbar:
                tqdm.write(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, KNN val acc={knn_acc_val:.4f}, Kernel val acc={kernel_acc_val:.4f}, Softmax val acc ={gcn_acc_val:.4f}")
        
            knn_acc_list.append(knn_acc_val)
            kernel_acc_list.append(kernel_acc_val)
            gcn_acc_list.append(gcn_acc_val)


            # knn_train_acc_list.append(knn_acc_train)

            if knn_acc_val > best_val_score:
                best_val_score = knn_acc_val
                # model_copy = copy.copy(model) 
                # model_copy = model
                # print("model replaced")
                gcn_model_copy = copy.deepcopy(model.data2vec) 
                theta_copy = model.theta.clone()

                # best_test_acc, _, _ = eval.test(model, data, K=K, h=1e-2, train_test_val = "test",
                #                                                 s_train_mask = train_mask, s_test_mask = test_mask)
        

    # print(best_test_acc)
    best_model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = gcn_model_copy, theta = theta_copy).to(device)
    best_knn_test_acc, best_kernel_acc_val, best_softmax_acc_val, _ = eval.test(best_model, data, K=K, h=h, train_test_val = "test",
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    print("best_knn_test_acc", best_knn_test_acc)
    print("best_kernel_acc_val", best_kernel_acc_val)
    # print(loss_list)
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([knn_acc_list], gap = gap, legend_list = ["knn_acc_list"])

    # ------------------------------ CLS training ------------------------------ #
    print("# ------------------------------ CLS training ------------------------------ #")

    # train nn to fit the dimension
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(NN_cls.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam([
                # {'params': model.parameters()},
                {'params': NN_cls.parameters()}
            ], lr=1e-3, weight_decay=5e-4)

    best_val_score = 0

    gap = 50
    batch_size = 20
    n_iter = 201

    loss_list = []
    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    gcn_acc_list = []

    for epoch in tqdm(range(n_iter)): 
        NN_cls.train()
        # model.eval()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            # print(Y_train.shape)
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            # print(Y_train.shape)
            # Q = sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            # Q = Q.to(device)
            # p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
            #             sorted_index = sorted_index.squeeze())
            # loss = F.nll_loss(out, Y_train.reshape((1,35)))
            # TV Loss
            # loss  = celoss(p_hat)

            # cls_.eval()
            # model.eval()
            
            # with torch.no_grad():
            # all training labels are used to generate the emprical dist.
            # Q       = sortedY2Q(Y_train.reshape((1, n_class * 3)))      
            # Q       = Q.to(device) 
            with torch.no_grad():
                H_train  = torch.index_select(model.data2vec(data)[mask], 0, sorted_index)             # [n_test_sample, n_feature]
            # m = torch.nn.Softmax(dim=1)
            # p_hat_test = m(H_train.t())
            # print("phat_test shape", p_hat_test.shape)
            # # return p_hat_test
            # print("p_hat", p_hat.shape)
            # print("H_train t", H_train.t().shape)


            # --- perform testset pred --- #
            # estimate the probs
            # p_hat_sft = softmax_regressor(H_train, H_train, p_hat, h)  # [n_class, n_grid * n_grid]
            p_hat_sft = NN_cls(H_train)
            # softmax_pred            = p_hat_sft.argmax(dim=1)
            # print("softmax_pred", p_hat_sft.shape)
            # print("Y_train", Y_train.shape)

            loss += criterion(p_hat_sft, Y_train)
            
        loss /= batch_size
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item()/batch_size)

        # print(loss_list)
        # print(np.mean(loss_list))
        if epoch % gap == 0:
            knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(model, data, K=K, h=h, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)
            kernel_acc_list.append(kernel_acc_val)
            knn_acc_list.append(knn_acc_val)
            # knn_train_acc_list = []
        
            gcn_acc_list.append(gcn_acc_val)
    
            if gcn_acc_val > best_val_score:
                best_val_score = gcn_acc_val
                # _, _, best_test_score = eval.test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = test_mask)
                # _, _, best_test_score = test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = cls_)
                cls_model_copy = copy.deepcopy(NN_cls) 
    
    # print("best_test_score", best_test_score)
    knn_accuracy_test, kernel_accuracy_test, best_softmax_acc_test, simple_knn_acc_test = eval.test(model, data, K=K, h=h, train_test_val = "test", cls_model = cls_model_copy,
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([gcn_acc_list], gap = gap, legend_list = ["gcn_acc_val"])

    print("best gnn acc", best_softmax_acc_test)
    print("simple_knn_acc_test", simple_knn_acc_test)
    print("knn_accuracy_test", knn_accuracy_test)
    print("kernel_accuracy_test", kernel_accuracy_test)

    search_through(model, data, classes, sample_per_class, train_mask, test_mask, NN_cls, K, 1)

def add_noise_main():
    print("Running add_noise_main.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    utils.seed_everything(0)

    
    # model config
    sample_per_class    = 10                             # number of samples per class given
    dataset             = Planetoid(root='/tmp/Cora', name='Cora', num_train_per_class = sample_per_class, split = 'random')
    n_class             = 7                             # number classes given
    classes             = [i for i in range(n_class)]   # listing of classes
    in_feature          = dataset.num_node_features     # the input feature length
    n_feature           = 2                             # the gcn output embedding dim
    batch_class_size    = 9                             # used part of all samples of a class per mini-set
                                                        # should be smaller than sample_per_class
    n_sample            = n_class * batch_class_size    # per batch number of samples used for cvs optimization
    max_theta           = 1e-2                          # maximum wasserstain distance for samples within a class
    lr                  = 0.1                           # lr
    K                   = 14                            # k nearest
    gcn_hidden_dim      = 16
    # batch_size          = 10                            # per batch number of sets
    # n_iter              = 101
    # gap = 20

    # init model
    # model       = rc.RobustGraphClassifier(n_class, n_sample, n_feature, max_theta)

    # init data
    data = dataset[0]
    # print(type(data.x))
    print("mean of data", torch.mean(data.x).item())
    print("std of data", torch.std(data.x).item())

    data = dataloader.add_noise_to_data(data, 1)
    data = data.to(device)

    # ------------------------------ GCN pretraining ------------------------------ #
    print("# ------------------------------ GNN pretraining ------------------------------ #")

    # pretrained_gcn = SimpleGCN(in_feature = dataset.num_node_features, out_feature = n_feature,
    #                            hidden_size = 16)
    pretrained_gcn = nn.GCN(input_dim = dataset.num_node_features, hidden_dim = gcn_hidden_dim)
    # pretrained_gcn = nn.GAT(num_features = dataset.num_node_features, hidden_dim = 56,
    # num_layers = 2,
    # num_classes = 7,
    # dropout = 0.6, heads = 8)
    NN_cls = nn.NN_classifier(input_dim = 7, output_dim = 7, keepprob = 0.9, linear_hidden_size=512)
    log_softmax_layer = nn.ClassficationLayer()

    train_mask = np.isin(data.y.cpu(), classes) * data.train_mask.cpu().numpy()
    test_mask = np.isin(data.y.cpu(), classes) * data.test_mask.cpu().numpy()
    val_mask = np.isin(data.y.cpu(), classes) * data.val_mask.cpu().numpy()

    # pretrained_gcn = SimpleGCN(in_feature =  , out_feature = 7, return_intermediate_value = False).to(device)
    # data = dataset[0].to(device)
    optimizer = torch.optim.Adam([
                    {'params': pretrained_gcn.parameters()},
                    {'params': NN_cls.parameters()}
                ], lr=0.01, weight_decay=5e-4)

    pretrained_gcn.to(device)
    NN_cls.to(device)
    log_softmax_layer.to(device)

    for epoch in range(1000):
        pretrained_gcn.train()
        NN_cls.train()
        optimizer.zero_grad()
        out = log_softmax_layer(NN_cls(pretrained_gcn(data)))
        # print(pretrained_gcn(data))
        # print(out.shape) # output shape: [2708, 7]
        # print(out[train_mask].shape) # output shape: [35, 7]
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        # print(out[train_mask].shape)
        # print(data.y[train_mask].shape)
        loss.backward()
        optimizer.step()
    
    # testing
    pretrained_gcn.eval()
    pred = log_softmax_layer(NN_cls(pretrained_gcn(data))).argmax(dim=1)
    print(f'Pretrain GNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain GNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')
    knn = KNeighborsClassifier(n_neighbors=K)

    X = pretrained_gcn(data)[train_mask].cpu().detach().numpy()
    y = data.y[train_mask].cpu().detach().numpy()
    knn.fit(X, y)
    pred = torch.tensor(knn.predict(pretrained_gcn(data).cpu().detach().numpy()))
    print(f'Pretrain KNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain KNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')

    # ------------------------------ DR training ------------------------------ #
    print("# ------------------------------ DR training ------------------------------ #")

    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    loss_list = []
    best_val_score = 0
    gcn_acc_list = []

    gap = 20
    batch_size = 20
    num_epochs = 30
    dr_early_stopping = False

    model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = pretrained_gcn).to(device)
    # model = rc.RobustGraphClassifier(n_class, n_sample, in_feature, out_channel = 7, hidden_size = 16).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
    flag_insolvant = False
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            Q = utils.sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            Q = Q.to(device)
            try:
                p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
                            sorted_index = sorted_index.squeeze())
            
                # loss = F.nll_loss(out, Y_train.reshape((1,35)))
                # TV Loss
                loss += utils.tvloss(p_hat)
                # loss  = celoss(p_hat)
            except:
                print("Insolvant error! Skip and stop.")
                flag_insolvant = True
                break
        if flag_insolvant:
            break
        else:
            loss/=batch_size
            loss_list.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            
        # model.eval()
        # pred = model(data, Q).argmax(dim=1)
        # print("pred shape", pred.shape)
        # pred = pred.reshape(35)
        # correct = (pred == data.y[data.train_mask]).sum()
        # acc = int(correct) / int(data.test_mask.sum())


        # print(loss_list)
        # print(np.mean(loss_list))
        if dr_early_stopping:
            if epoch % gap == 0:
                # loss_list.append(loss.item()/batch_size)
                # knn_acc_val, kernel_acc_val = test(model, data, K=K, h=1e-2, train_test_val = "train")
                # kernel_acc_list.append(kernel_acc_val)
                # knn_acc_list.append(knn_acc_val)

                knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(model, data, K=K, h=1e-2, train_test_val = "val",
                                                                    s_train_mask = train_mask, s_test_mask = val_mask)
                # knn_acc_train, kernel_acc_train = test(model, data, K=K, h=1e-2, train_test_val = "train")

                # print("Train loss: {:.6f} \nVal set ({:d} samples):  kNN accuracy: {:.3f}, kernel smoothing accuracy: {:.3f}, Softmax acc: {:.3f} \n".format(loss.item()/batch_size, \
                #                                             sum(data.val_mask), knn_acc_val, kernel_acc_val, gcn_acc_val))
                # print("Train kNN accurary: %.3f\nTest set (%d samples):  kNN accuracy: %.3f, kernel smoothing accuracy: %.3f\n" 
                #       % (knn_accuracy_train, sum(data.test_mask), knn_accuracy, kernel_accuracy))
                
                with tqdm(total=1, disable=True) as pbar:
                    tqdm.write(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, KNN val acc={knn_acc_val:.4f}, Kernel val acc={kernel_acc_val:.4f}, Softmax val acc ={gcn_acc_val:.4f}")
            
                knn_acc_list.append(knn_acc_val)
                kernel_acc_list.append(kernel_acc_val)
                gcn_acc_list.append(gcn_acc_val)


                # knn_train_acc_list.append(knn_acc_train)

                if knn_acc_val > best_val_score:
                    best_val_score = knn_acc_val
                    # model_copy = copy.copy(model) 
                    # model_copy = model
                    # print("model replaced")
                    gcn_model_copy = copy.deepcopy(model.data2vec) 
                    theta_copy = model.theta.clone()

                    # best_test_acc, _, _ = eval.test(model, data, K=K, h=1e-2, train_test_val = "test",
                    #                                                 s_train_mask = train_mask, s_test_mask = test_mask)
    if not dr_early_stopping:
        gcn_model_copy = copy.deepcopy(model.data2vec) 
        theta_copy = model.theta.clone()

    # print(best_test_acc)
    best_model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = gcn_model_copy, theta = theta_copy).to(device)
    best_knn_test_acc, best_kernel_acc_val, best_softmax_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "test",
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    print("best_knn_test_acc", best_knn_test_acc)
    # print(loss_list)
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([knn_acc_list], gap = gap, legend_list = ["knn_acc_list"])

    # ------------------------------ CLS training ------------------------------ #
    print("# ------------------------------ CLS training ------------------------------ #")

    # train nn to fit the dimension
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(NN_cls.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam([
                # {'params': model.parameters()},
                {'params': NN_cls.parameters()}
            ], lr=1e-3, weight_decay=5e-4)
    
    H_val, Y_val = eval.get_hidden_state(best_model, data, K=K, h=1e-2, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)

    best_val_score = 0

    gap = 20
    batch_size = 20
    n_iter = 501

    loss_list = []
    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    gcn_acc_list = []

    for epoch in tqdm(range(n_iter)):
        NN_cls.train()
        # model.eval()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            # print(Y_train.shape)
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            # print(Y_train.shape)
            # Q = sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            # Q = Q.to(device)
            # p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
            #             sorted_index = sorted_index.squeeze())
            # loss = F.nll_loss(out, Y_train.reshape((1,35)))
            # TV Loss
            # loss  = celoss(p_hat)

            # cls_.eval()
            # model.eval()
            
            # with torch.no_grad():
            # all training labels are used to generate the emprical dist.
            # Q       = sortedY2Q(Y_train.reshape((1, n_class * 3)))      
            # Q       = Q.to(device) 
            with torch.no_grad():
                H_train  = torch.index_select(best_model.data2vec(data)[mask], 0, sorted_index)             # [n_test_sample, n_feature]
            # m = torch.nn.Softmax(dim=1)
            # p_hat_test = m(H_train.t())
            # print("phat_test shape", p_hat_test.shape)
            # # return p_hat_test
            # print("p_hat", p_hat.shape)
            # print("H_train t", H_train.t().shape)


            # --- perform testset pred --- #
            # estimate the probs
            # p_hat_sft = softmax_regressor(H_train, H_train, p_hat, h)  # [n_class, n_grid * n_grid]
            p_hat_sft = NN_cls(H_train)
            # softmax_pred            = p_hat_sft.argmax(dim=1)
            # print("softmax_pred", p_hat_sft.shape)
            # print("Y_train", Y_train.shape)

            loss += criterion(p_hat_sft, Y_train)
            
        loss /= batch_size
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item()/batch_size)

        # print(loss_list)
        # print(np.mean(loss_list))
        if epoch % gap == 0:
            # knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)
            gcn_acc_val = eval.test_cls_model(NN_cls, H_val, Y_val)
            # kernel_acc_list.append(kernel_acc_val)
            # knn_acc_list.append(knn_acc_val)
            # knn_train_acc_list = []
        
            gcn_acc_list.append(gcn_acc_val)
            with tqdm(total=1, disable=True) as pbar:
                tqdm.write(f"Epoch {epoch+1}/{n_iter}: Loss={loss:.4f}, gcn_acc_val acc ={gcn_acc_val:.4f}")
                # tqdm.write(f"Epoch {epoch+1}/{n_iter}: Loss={loss:.4f}, KNN val acc={knn_acc_val:.4f}, Kernel val acc={kernel_acc_val:.4f}, Softmax val acc ={gcn_acc_val:.4f}")
        
            if gcn_acc_val > best_val_score:
                best_val_score = gcn_acc_val
                # _, _, best_test_score = eval.test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = test_mask)
                # _, _, best_test_score = test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = cls_)
                cls_model_copy = copy.deepcopy(NN_cls) 
    
    # print("best_test_score", best_test_score)
    _, _, best_softmax_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "test", cls_model = cls_model_copy,
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([gcn_acc_list], gap = gap, legend_list = ["gcn_acc_val"])

    print("best_softmax_acc_val", best_softmax_acc_val)

def remove_edge_main():
    print("Running add_noise_main.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    utils.seed_everything(0)

    
    # model config
    sample_per_class    = 10                             # number of samples per class given
    dataset             = Planetoid(root='/tmp/Cora', name='Cora', num_train_per_class = sample_per_class, split = 'random')
    n_class             = 7                             # number classes given
    classes             = [i for i in range(n_class)]   # listing of classes
    in_feature          = dataset.num_node_features     # the input feature length
    n_feature           = 2                             # the gcn output embedding dim
    batch_class_size    = 9                             # used part of all samples of a class per mini-set
                                                        # should be smaller than sample_per_class
    n_sample            = n_class * batch_class_size    # per batch number of samples used for cvs optimization
    max_theta           = 1e-2                          # maximum wasserstain distance for samples within a class
    lr                  = 0.1                           # lr
    K                   = 14                            # k nearest
    gcn_hidden_dim      = 16
    # batch_size          = 10                            # per batch number of sets
    # n_iter              = 101
    # gap = 20

    # init model
    # model       = rc.RobustGraphClassifier(n_class, n_sample, n_feature, max_theta)

    # init data
    data = dataset[0]
    # print(type(data.x))
    print("mean of data", torch.mean(data.x).item())
    print("std of data", torch.std(data.x).item())

    data = dataloader.remove_edge_to_data(data, 0.2)
    data = data.to(device)

    # ------------------------------ GCN pretraining ------------------------------ #
    print("# ------------------------------ GNN pretraining ------------------------------ #")

    # pretrained_gcn = SimpleGCN(in_feature = dataset.num_node_features, out_feature = n_feature,
    #                            hidden_size = 16)
    pretrained_gcn = nn.GCN(input_dim = dataset.num_node_features, hidden_dim = gcn_hidden_dim)
    # pretrained_gcn = nn.GAT(num_features = dataset.num_node_features, hidden_dim = 56,
    # num_layers = 2,
    # num_classes = 7,
    # dropout = 0.6, heads = 8)
    NN_cls = nn.NN_classifier(input_dim = 7, output_dim = 7, keepprob = 0.9, linear_hidden_size=512)
    log_softmax_layer = nn.ClassficationLayer()

    train_mask = np.isin(data.y.cpu(), classes) * data.train_mask.cpu().numpy()
    test_mask = np.isin(data.y.cpu(), classes) * data.test_mask.cpu().numpy()
    val_mask = np.isin(data.y.cpu(), classes) * data.val_mask.cpu().numpy()

    # pretrained_gcn = SimpleGCN(in_feature =  , out_feature = 7, return_intermediate_value = False).to(device)
    # data = dataset[0].to(device)
    optimizer = torch.optim.Adam([
                    {'params': pretrained_gcn.parameters()},
                    {'params': NN_cls.parameters()}
                ], lr=0.01, weight_decay=5e-4)

    pretrained_gcn.to(device)
    NN_cls.to(device)
    log_softmax_layer.to(device)

    for epoch in range(1000):
        pretrained_gcn.train()
        NN_cls.train()
        optimizer.zero_grad()
        out = log_softmax_layer(NN_cls(pretrained_gcn(data)))
        # print(pretrained_gcn(data))
        # print(out.shape) # output shape: [2708, 7]
        # print(out[train_mask].shape) # output shape: [35, 7]
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        # print(out[train_mask].shape)
        # print(data.y[train_mask].shape)
        loss.backward()
        optimizer.step()
    
    # testing
    pretrained_gcn.eval()
    pred = log_softmax_layer(NN_cls(pretrained_gcn(data))).argmax(dim=1)
    print(f'Pretrain GNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain GNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')
    knn = KNeighborsClassifier(n_neighbors=K)

    X = pretrained_gcn(data)[train_mask].cpu().detach().numpy()
    y = data.y[train_mask].cpu().detach().numpy()
    knn.fit(X, y)
    pred = torch.tensor(knn.predict(pretrained_gcn(data).cpu().detach().numpy()))
    print(f'Pretrain KNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain KNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')

    # ------------------------------ DR training ------------------------------ #
    print("# ------------------------------ DR training ------------------------------ #")

    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    loss_list = []
    best_val_score = 0
    gcn_acc_list = []

    gap = 20
    batch_size = 20
    num_epochs = 30
    dr_early_stopping = False

    model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = pretrained_gcn).to(device)
    # model = rc.RobustGraphClassifier(n_class, n_sample, in_feature, out_channel = 7, hidden_size = 16).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
    flag_insolvant = False
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            Q = utils.sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            Q = Q.to(device)
            try:
                p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
                            sorted_index = sorted_index.squeeze())
            
                # loss = F.nll_loss(out, Y_train.reshape((1,35)))
                # TV Loss
                loss += utils.tvloss(p_hat)
                # loss  = celoss(p_hat)
            except:
                print("Insolvant error! Skip and stop.")
                flag_insolvant = True
                break
        if flag_insolvant:
            break
        else:
            loss/=batch_size
            loss_list.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            
        # model.eval()
        # pred = model(data, Q).argmax(dim=1)
        # print("pred shape", pred.shape)
        # pred = pred.reshape(35)
        # correct = (pred == data.y[data.train_mask]).sum()
        # acc = int(correct) / int(data.test_mask.sum())


        # print(loss_list)
        # print(np.mean(loss_list))
        if dr_early_stopping:
            if epoch % gap == 0:
                # loss_list.append(loss.item()/batch_size)
                # knn_acc_val, kernel_acc_val = test(model, data, K=K, h=1e-2, train_test_val = "train")
                # kernel_acc_list.append(kernel_acc_val)
                # knn_acc_list.append(knn_acc_val)

                knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(model, data, K=K, h=1e-2, train_test_val = "val",
                                                                    s_train_mask = train_mask, s_test_mask = val_mask)
                # knn_acc_train, kernel_acc_train = test(model, data, K=K, h=1e-2, train_test_val = "train")

                # print("Train loss: {:.6f} \nVal set ({:d} samples):  kNN accuracy: {:.3f}, kernel smoothing accuracy: {:.3f}, Softmax acc: {:.3f} \n".format(loss.item()/batch_size, \
                #                                             sum(data.val_mask), knn_acc_val, kernel_acc_val, gcn_acc_val))
                # print("Train kNN accurary: %.3f\nTest set (%d samples):  kNN accuracy: %.3f, kernel smoothing accuracy: %.3f\n" 
                #       % (knn_accuracy_train, sum(data.test_mask), knn_accuracy, kernel_accuracy))
                
                with tqdm(total=1, disable=True) as pbar:
                    tqdm.write(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, KNN val acc={knn_acc_val:.4f}, Kernel val acc={kernel_acc_val:.4f}, Softmax val acc ={gcn_acc_val:.4f}")
            
                knn_acc_list.append(knn_acc_val)
                kernel_acc_list.append(kernel_acc_val)
                gcn_acc_list.append(gcn_acc_val)


                # knn_train_acc_list.append(knn_acc_train)

                if knn_acc_val > best_val_score:
                    best_val_score = knn_acc_val
                    # model_copy = copy.copy(model) 
                    # model_copy = model
                    # print("model replaced")
                    gcn_model_copy = copy.deepcopy(model.data2vec) 
                    theta_copy = model.theta.clone()

                    # best_test_acc, _, _ = eval.test(model, data, K=K, h=1e-2, train_test_val = "test",
                    #                                                 s_train_mask = train_mask, s_test_mask = test_mask)
    if not dr_early_stopping:
        gcn_model_copy = copy.deepcopy(model.data2vec) 
        theta_copy = model.theta.clone()

    # print(best_test_acc)
    best_model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = gcn_model_copy, theta = theta_copy).to(device)
    best_knn_test_acc, best_kernel_acc_val, best_softmax_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "test",
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    print("best_knn_test_acc", best_knn_test_acc)
    # print(loss_list)
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([knn_acc_list], gap = gap, legend_list = ["knn_acc_list"])

    # ------------------------------ CLS training ------------------------------ #
    print("# ------------------------------ CLS training ------------------------------ #")

    # train nn to fit the dimension
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(NN_cls.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam([
                # {'params': model.parameters()},
                {'params': NN_cls.parameters()}
            ], lr=1e-3, weight_decay=5e-4)
    
    H_val, Y_val = eval.get_hidden_state(best_model, data, K=K, h=1e-2, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)

    best_val_score = 0

    gap = 20
    batch_size = 20
    n_iter = 501

    loss_list = []
    kernel_acc_list = []
    knn_acc_list = []
    knn_train_acc_list = []
    gcn_acc_list = []

    for epoch in tqdm(range(n_iter)):
        NN_cls.train()
        # model.eval()
        optimizer.zero_grad()

        # Q = sortedY2Q(Y)      # calculate empirical distribution based on labels
        # true_y = data.y[data.train_mask]
        loss = 0
        
        for i in range(batch_size):
            # -- batching -- #
            temp_mask = dataloader.get_train_mask(batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            # print(Y_train.shape)
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            # print(Y_train.shape)
            # Q = sortedY2Q(Y_train.reshape((1, batch_class_size * n_class)))
            # Q = Q.to(device)
            # p_hat = model(data, Q, train_flag = True, minibatch_mask = mask, 
            #             sorted_index = sorted_index.squeeze())
            # loss = F.nll_loss(out, Y_train.reshape((1,35)))
            # TV Loss
            # loss  = celoss(p_hat)

            # cls_.eval()
            # model.eval()
            
            # with torch.no_grad():
            # all training labels are used to generate the emprical dist.
            # Q       = sortedY2Q(Y_train.reshape((1, n_class * 3)))      
            # Q       = Q.to(device) 
            with torch.no_grad():
                H_train  = torch.index_select(best_model.data2vec(data)[mask], 0, sorted_index)             # [n_test_sample, n_feature]
            # m = torch.nn.Softmax(dim=1)
            # p_hat_test = m(H_train.t())
            # print("phat_test shape", p_hat_test.shape)
            # # return p_hat_test
            # print("p_hat", p_hat.shape)
            # print("H_train t", H_train.t().shape)


            # --- perform testset pred --- #
            # estimate the probs
            # p_hat_sft = softmax_regressor(H_train, H_train, p_hat, h)  # [n_class, n_grid * n_grid]
            p_hat_sft = NN_cls(H_train)
            # softmax_pred            = p_hat_sft.argmax(dim=1)
            # print("softmax_pred", p_hat_sft.shape)
            # print("Y_train", Y_train.shape)

            loss += criterion(p_hat_sft, Y_train)
            
        loss /= batch_size
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item()/batch_size)

        # print(loss_list)
        # print(np.mean(loss_list))
        if epoch % gap == 0:
            # knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)
            gcn_acc_val = eval.test_cls_model(NN_cls, H_val, Y_val)
            # kernel_acc_list.append(kernel_acc_val)
            # knn_acc_list.append(knn_acc_val)
            # knn_train_acc_list = []
        
            gcn_acc_list.append(gcn_acc_val)
            with tqdm(total=1, disable=True) as pbar:
                tqdm.write(f"Epoch {epoch+1}/{n_iter}: Loss={loss:.4f}, gcn_acc_val acc ={gcn_acc_val:.4f}")
                # tqdm.write(f"Epoch {epoch+1}/{n_iter}: Loss={loss:.4f}, KNN val acc={knn_acc_val:.4f}, Kernel val acc={kernel_acc_val:.4f}, Softmax val acc ={gcn_acc_val:.4f}")
        
            if gcn_acc_val > best_val_score:
                best_val_score = gcn_acc_val
                # _, _, best_test_score = eval.test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = test_mask)
                # _, _, best_test_score = test(model, data, K=K, h=1e-2, train_test_val = "test", cls_model = cls_)
                cls_model_copy = copy.deepcopy(NN_cls) 
    
    # print("best_test_score", best_test_score)
    _, _, best_softmax_acc_test, _ = eval.test(best_model, data, K=K, h=1e-2, train_test_val = "test", cls_model = cls_model_copy,
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"])
    plot_with_dist([gcn_acc_list], gap = gap, legend_list = ["gcn_acc_val"])

    print("best_softmax_acc_test", best_softmax_acc_test)

def gcn_vanilla_main(args):
    '''
    Main function for GCN / GAT + NN vanilla.
    '''
    print("Running gcn_vanilla_main.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    utils.seed_everything(args.seed)

    # Model config
    sample_per_class    = args.sample_per_class                             # number of samples per class given
    dataset             = Planetoid(root='/tmp/' + args.dataset, name=args.dataset, num_train_per_class = sample_per_class, split = 'random')
    n_class             = args.n_class                             # number classes given
    classes             = [i for i in range(n_class)]   # listing of classes
    in_feature          = dataset.num_node_features     # the input feature length
    n_feature           = args.n_feature                             # the gcn output embedding dim
    gnn_hidden_dim      = args.gnn_hidden_dim
    # batch_size          = 10                            # per batch number of sets
    # n_iter              = 101
    # gap = 20

        # Init model
    if args.gnn_type == 'gcn':
        num_layers = 2
        model = nn.GCN(in_feature, gnn_hidden_dim, n_feature, num_layers).to(device)
    elif args.gnn_type == 'gat':
        hidden_dim = 16
        heads = 16
        dropout = 0.6
        num_layers = 2
        model = nn.GAT(
            in_feature, hidden_dim, n_feature, num_layers, heads, dropout
        ).to(device)

    # Init data
    data = dataset[0].to(device)

    # GCN pretraining
    NN_cls = nn.NN_classifier(
        input_dim=n_feature, output_dim=len(classes), keepprob=0.9,
        linear_hidden_size=1024
    )

    train_mask = np.isin(data.y.cpu(), classes) * data.train_mask.cpu().numpy()
    test_mask = np.isin(data.y.cpu(), classes) * data.test_mask.cpu().numpy()
    val_mask = np.isin(data.y.cpu(), classes) * data.val_mask.cpu().numpy()

    # Train GNN and NN classifier
    best_model, best_nn_cls = gnn_training(
        dataset, model, NN_cls, data, train_mask, val_mask, test_mask, args, device
    )


def exp_main(args):
    '''
    Main function for GCN / GAT + NN test under noise attack.
    '''
    print("Running experiment main.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    utils.seed_everything(args.seed)

    # Model config
    dataset             = Planetoid(root='/tmp/' + args.dataset, name=args.dataset, num_train_per_class = args.sample_per_class, split = 'random')
    classes             = [i for i in range(args.n_class)]   # listing of classes
    in_feature          = dataset.num_node_features     # the input feature length
    # batch_size          = 10                            # per batch number of sets
    # n_iter              = 101
    # gap = 20

    # Init model
    if args.gnn_type == 'gcn':
        model = nn.GCN(in_feature, args.gnn_hidden_dim, args.n_feature, args.gnn_hidden_layers).to(device)
    elif args.gnn_type == 'gat':
        heads = 16
        dropout = 0.6
        model = nn.GAT(
            in_feature, args.gnn_hidden_dim, args.n_feature, args.gnn_hidden_layers, heads, dropout
        ).to(device)

    # Init data
    data = dataset[0].to(device)
    data = dataloader.add_noise_to_data(data, pct = args.noise_rate)
    data = dataloader.remove_edge_to_data(data, drop_rate = args.edge_removel_rate)


    # GCN pretraining
    NN_cls = nn.NN_classifier(
        input_dim=args.n_feature, output_dim=len(classes), keepprob=0.9,
        linear_hidden_size=1024
    )

    train_mask = np.isin(data.y.cpu(), classes) * data.train_mask.cpu().numpy()
    test_mask = np.isin(data.y.cpu(), classes) * data.test_mask.cpu().numpy()
    val_mask = np.isin(data.y.cpu(), classes) * data.val_mask.cpu().numpy()

    # Step 1. Train GNN and NN classifier
    best_gnn_model, best_nn_cls = gnn_training(
        dataset, model, NN_cls, data, train_mask, val_mask, test_mask, args, device
    )

    # Step 2. DR Training GNN embedding 
    best_gnn_model = drgnn_training(dataset, data, classes, best_gnn_model, train_mask, val_mask, test_mask, args, device)

    # Step 3. Retrain NN classifier
    nn_cls_training(dataset, data, classes, best_gnn_model, best_nn_cls, train_mask, val_mask, test_mask, args, device)



if __name__ == "__main__":
    # plot_main()
    # drgcn_exp_main()
    # add_noise_main()
    # remove_edge_main()

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora',
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=42,
                        help='')
    parser.add_argument('--sample_per_class', type=int, default=10,
                        help='number of samples per class given')
    parser.add_argument('--n_class', type=int, default=7,
                        help='number of classes given')
    parser.add_argument('--n_feature', type=int, default=2,
                        help='the gcn output embedding dim')
    parser.add_argument('--batch_class_size', type=int, default=4,
                        help='used part of all samples of a class per mini-set, should be smaller than sample_per_class')
    parser.add_argument('--max_theta', type=float, default=1e-2,
                        help='maximum wasserstain distance for samples within a class')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--K', type=int, default=14,
                        help='k nearest')
    parser.add_argument('--h', type=float, default=1e-2,
                        help='k nearest')
    parser.add_argument('--gnn_type', type=str,
                    help='gcn or gat', default = "gcn")
    parser.add_argument('--gnn_hidden_dim', type=int, default=16,
                        help='hidden dim of GCN')
    parser.add_argument('--gnn_hidden_layers', type=int, default=2,
                        help='hidden dim of GCN')
    parser.add_argument('--gnn_early_stopping', type=bool,
                help='', default = False)
    parser.add_argument('--dr_epochs', type=int,
                help='', default = 501)
    parser.add_argument('--dr_early_stopping', type=bool,
                help='', default = True)
    parser.add_argument('--edge_removel_rate', type=float,
                help='', default = 0)
    parser.add_argument('--noise_rate', type=float,
                help='', default = 1)
    parser.add_argument('--exp_name', type=str,
                    help='experiment name', default = "")

    args = parser.parse_args()
    print('Args info:')
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    
    # gcn_vanilla_main(args)
    exp_main(args)
