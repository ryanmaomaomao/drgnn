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

def gnn_training(dataset, pretrained_gcn, NN_cls, data, train_mask, val_mask, test_mask, args, device):
    # ------------------------------ GCN pretraining ------------------------------ #
    print("# ------------------------------ GNN pretraining ------------------------------ #")

    # pretrained_gcn = SimpleGCN(in_feature = dataset.num_node_features, out_feature = n_feature,
    #                            hidden_size = 16)
    # pretrained_gcn = nn.GAT(num_features = dataset.num_node_features, hidden_dim = 56,
    # num_layers = 2,
    # num_classes = 7,
    # dropout = 0.6, heads = 8)
    log_softmax_layer = nn.ClassficationLayer()


    # pretrained_gcn = SimpleGCN(in_feature =  , out_feature = 7, return_intermediate_value = False).to(device)
    # data = dataset[0].to(device)
    optimizer = torch.optim.Adam([
                    {'params': pretrained_gcn.parameters()},
                    {'params': NN_cls.parameters()}
                ], lr=0.001, weight_decay=5e-4)

    pretrained_gcn.to(device)
    NN_cls.to(device)
    log_softmax_layer.to(device)
    best_gnn_val = 0

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

        if args.gnn_early_stopping:

            # testing
            pretrained_gcn.eval()
            NN_cls.eval()

            pred = out.argmax(dim=1)
            val_score = eval.get_acc_for_mask(pred, data, val_mask)
            if val_score > best_gnn_val:
                best_gnn_val = val_score
                best_model = copy.deepcopy(pretrained_gcn)
                best_nn_cls = copy.deepcopy(NN_cls)

    if not args.gnn_early_stopping:
        best_model = pretrained_gcn
        best_nn_cls = NN_cls

    # testing
    best_model.eval()
    best_nn_cls.eval()
    pred = log_softmax_layer(best_nn_cls(best_model(data))).argmax(dim=1)
    print(f'Pretrain GNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain GNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')
    knn = KNeighborsClassifier(n_neighbors=args.K)

    X = best_model(data)[train_mask].cpu().detach().numpy()
    y = data.y[train_mask].cpu().detach().numpy()
    knn.fit(X, y)
    pred = torch.tensor(knn.predict(best_model(data).cpu().detach().numpy()))
    print(f'Pretrain KNN Val Acc: {eval.get_acc_for_mask(pred, data, val_mask):.4f}')
    print(f'Pretrain KNN Test Acc: {eval.get_acc_for_mask(pred, data, test_mask):.4f}')

    return best_model, best_nn_cls

def drgnn_training(dataset, data, classes, pretrained_gcn, train_mask, val_mask, test_mask, args, device):
    n_class = len(classes)
    n_sample = args.n_class * args.batch_class_size
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
    num_epochs = args.dr_epochs

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
            temp_mask = dataloader.get_train_mask(args.batch_class_size, dataset, classes)
            mask = torch.tensor(temp_mask)
            Y_train = data.y[mask]
            # -------------- #
            sorted_index = torch.argsort(Y_train)
            Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())
            Q = utils.sortedY2Q(Y_train.reshape((1, args.batch_class_size * n_class)))
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
        if args.dr_early_stopping:
            if epoch % gap == 0:
                # loss_list.append(loss.item()/batch_size)
                # knn_acc_val, kernel_acc_val = test(model, data, K=K, h=h, train_test_val = "train")
                # kernel_acc_list.append(kernel_acc_val)
                # knn_acc_list.append(knn_acc_val)

                knn_acc_val, kernel_acc_val, gcn_acc_val, _ = eval.test(model, data, K=args.K, h=args.h, train_test_val = "val",
                                                                    s_train_mask = train_mask, s_test_mask = val_mask)
                # knn_acc_train, kernel_acc_train = test(model, data, K=K, h=h, train_test_val = "train")

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

                    # best_test_acc, _, _ = eval.test(model, data, K=K, h=h, train_test_val = "test",
                    #                                                 s_train_mask = train_mask, s_test_mask = test_mask)
    if not args.dr_early_stopping:
        gcn_model_copy = copy.deepcopy(model.data2vec) 
        theta_copy = model.theta.clone()

    # print(best_test_acc)
    best_model = rc.RobustGraphClassifier(n_class = len(classes), n_sample = n_sample, in_channel = 2, out_channel = len(classes), gcn_pre_trained = gcn_model_copy, theta = theta_copy).to(device)
    best_knn_test_acc, best_kernel_acc_val, best_softmax_acc_val, _ = eval.test(best_model, data, K=args.K, h=args.h, train_test_val = "test",
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    print("best_knn_test_acc", best_knn_test_acc)
    # print(loss_list)
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"], exp_name=args.exp_name + "_dr_loss_list")
    plot_with_dist([knn_acc_list], gap = gap, legend_list = ["knn_acc_list"], exp_name=args.exp_name + "_dr_knn_acc_list")

    return best_model

def nn_cls_training(dataset, data, classes, best_model, NN_cls, train_mask, val_mask, test_mask, args, device):
    # ------------------------------ CLS training ------------------------------ #
    print("# ------------------------------ CLS training ------------------------------ #")

    # train nn to fit the dimension
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(NN_cls.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.Adam([
                # {'params': model.parameters()},
                {'params': NN_cls.parameters()}
            ], lr=1e-3, weight_decay=5e-4)
    
    H_val, Y_val = eval.get_hidden_state(best_model, data, K=args.K, h=1e-2, train_test_val = "val", cls_model = NN_cls, s_train_mask=train_mask, s_test_mask = val_mask)

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
            temp_mask = dataloader.get_train_mask(args.batch_class_size, dataset, classes)
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
    _, _, best_softmax_acc_val, _ = eval.test(best_model, data, K=args.K, h=1e-2, train_test_val = "test", cls_model = cls_model_copy,
                                                                s_train_mask = train_mask, s_test_mask = test_mask)
    
    plot_with_dist([loss_list], gap = gap, legend_list = ["loss_list"], exp_name=args.exp_name + "_nn_cls_loss_list")
    plot_with_dist([gcn_acc_list], gap = gap, legend_list = ["gcn_acc_val"],exp_name=args.exp_name + "_nn_cls_gcn_acc_list")

    print("best_softmax_acc_val", best_softmax_acc_val)