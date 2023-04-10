color_set = ["blue", "red", "green", "grey"]
cmap_set  = ["Blues", "Reds", "Greens", "Greys"]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import robust_classifier as rc
import torch
import nn as nn
import utils
from eval import get_acc_for_mask
from numpy.ma import masked_array
from sklearn.manifold import TSNE
from eval import knn_regressor, kernel_smoother, simple_knn_regressor

def plot_with_dist(acc_lists, gap = 1, title = "", legend_list = [], exp_name = "default"):
    # Plot Acc.
    epoch_range = [i * gap for i in range(len(acc_lists[0]))]
    color = cm.rainbow(np.linspace(0, 1, len(acc_lists)))
    # epoch_range = val_epoch_list
    for i in range(len(acc_lists)):
        plt.plot(epoch_range, acc_lists[i], label=legend_list[i], c = color[i])
    plt.title(title)
    plt.xlabel("Epoches")
    plt.legend()
    # plt.show()
    # plt.plot(acc_list)

    # plt.axis('off')
    # plt.savefig("results/%s_map_%s.pdf" % (prefix, arrow.now()), bbox_inches='tight')
    plt.savefig('images/' + exp_name + '.pdf')
    plt.clf()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """truncate colormap by proportion"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def visualize_2Dspace_Nclass(
    n_grid, max_H, min_H, p_hat_test, 
    H_train, Y_train, H_test, Y_test, classes, K, h, prefix="test"):
    """
    visualize 2D embedding space and corresponding training data points.
    """
    n_class = p_hat_test.shape[0]
    assert n_grid * n_grid == p_hat_test.shape[1]
    n_train_sample = H_train.shape[0]
    n_test_sample  = H_test.shape[0]
    # organize the p_hat as multiple matrices for each class
    p_hat_test = p_hat_test.numpy().reshape(n_class, n_grid, n_grid)
    p_hat_max  = p_hat_test.argmax(0) # [n_grid, n_grid]
    p_hat_mats = []                   # (n_class [n_grid, n_grid]) 
    for i in range(n_class):
        p_hat_show = p_hat_test[i] / p_hat_test.sum(0)
        p_hat_mat  = masked_array(p_hat_show, p_hat_max != i)
        p_hat_mats.append(p_hat_mat)
    # scale the training data to (0, n_grid)
    H_train = (H_train - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_train_sample, axis=0).reshape(-1,2)
    H_train = np.nan_to_num(H_train) * n_grid
    # scale the testing data to (0, n_grid)
    H_test = (H_test - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_test_sample, axis=0).reshape(-1,2)
    H_test = np.nan_to_num(H_test) * n_grid
    # prepare label set
    Y_set     = list(set(Y_train))
    Y_set.sort()
    # plot the region
    fig, ax = plt.subplots(1, 1)
    cmaps   = [ 
        truncate_colormap(cm.get_cmap(cmap), 0., 0.7) 
        for cmap in cmap_set[:n_class] ]
    # print(cmaps)
    implots = [ 
        ax.imshow(p_hat_mats[i], vmin=p_hat_mats[i].min(), vmax=p_hat_mats[i].max(), cmap=cmaps[i]) 
        for i in range(n_class) ]
    
    # plot the points
    # print("H_train shape", H_train.shape)
    # print("Y_test", len(Y_test))
    # print("Y_train", len(Y_train))
    for c, y in zip(color_set[:n_class], Y_set):
        # print("color", c)
        # print("Y label", y)
        Y_train_inds = np.where(Y_train == y)[0]
        # print("Y_train_inds", len(Y_train_inds))
        Y_test_inds  = np.where(Y_test == y)[0]
        plt.scatter(H_train[Y_train_inds, 1], H_train[Y_train_inds, 0], 
            s=30, c=c, linewidths=1, edgecolors="black")
        plt.scatter(H_test[Y_test_inds, 1], H_test[Y_test_inds, 0], 
            s=5, c=c, alpha=0.3)
    plt.axis('off')
    plt.show()


    for j in classes:
        fig, ax = plt.subplots(1, 1)
        implots = [ 
            ax.imshow(p_hat_mats[i], vmin=p_hat_mats[i].min(), vmax=p_hat_mats[i].max(), cmap=cmaps[i]) 
            for i in [j] ]
        # plot the points
        for c, y in zip(color_set[:n_class], Y_set):
            Y_train_inds = np.where(Y_train == y)[0]
            Y_test_inds  = np.where(Y_test == y)[0]
            plt.scatter(H_train[Y_train_inds, 1], H_train[Y_train_inds, 0], 
                s=30, c=c, linewidths=1, edgecolors="black")
            # plt.scatter(H_test[Y_test_inds, 1], H_test[Y_test_inds, 0], 
            #     s=5, c=c, alpha=0.3)
        plt.axis('off')
        plt.show()
    # plt.savefig("results/%s_map_%s.pdf" % (prefix, arrow.now()), bbox_inches='tight')
    plt.clf()
    
    for j in classes:
        fig, ax = plt.subplots(1, 1)
        implots = [ 
            ax.imshow(p_hat_test[i] / p_hat_test.sum(0), vmin=p_hat_mats[i].min(), vmax=p_hat_mats[i].max(), cmap=cmaps[i]) 
            for i in [j] ]
        # plot the points
        for c, y in zip(color_set[:n_class], Y_set):
            Y_train_inds = np.where(Y_train == y)[0]
            Y_test_inds  = np.where(Y_test == y)[0]
            plt.scatter(H_train[Y_train_inds, 1], H_train[Y_train_inds, 0], 
                s=30, c=c, linewidths=1, edgecolors="black")
            # plt.scatter(H_test[Y_test_inds, 1], H_test[Y_test_inds, 0], 
            #     s=5, c=c, alpha=0.3)
        plt.axis('off')
        plt.show()
    # plt.savefig("results/%s_map_%s.pdf" % (prefix, arrow.now()), bbox_inches='tight')
    plt.clf()

# given hidden embedding, evaluate corresponding p_hat 
# using the output of the robust classifier layer
def evaluate_p_hat(H, Q, theta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_class, n_sample, n_feature = theta.shape[1], H.shape[1], H.shape[2]
    rbstclf = rc.RobustClassifierLayer(n_class, n_sample, n_feature).to(device)
    return rbstclf(H, Q, theta).data

def search_through(model, data, classes, sample_per_class, selected_train_mask, selected_test_mask, NN_cls = None, K = 2, h = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_class = len(classes)

    Y_train = data.y[selected_train_mask]
    sorted_index_tr = torch.argsort(Y_train).squeeze()
    Y_train = torch.index_select(Y_train, 0, sorted_index_tr)
    # print("Y_train", Y_train.shape)

    Y_test = data.y[selected_test_mask]
    sorted_index_test = torch.argsort(Y_test).squeeze()
    Y_test = torch.index_select(Y_test, 0, sorted_index_test)

    with torch.no_grad():
        Q = utils.sortedY2Q(Y_train.reshape((1, n_class * sample_per_class))).to(device)               # [1, n_class, n_sample]
        H_train = torch.index_select(model.data2vec(data)[selected_train_mask], 0, sorted_index_tr)                 # [n_train_sample, n_feature]
        H_test  = torch.index_select(model.data2vec(data)[selected_test_mask], 0, sorted_index_test)                  # [n_test_sample, n_feature]
        theta   = model.theta.data.unsqueeze(0)           # [1, n_class]
        # print(type(H_train.unsqueeze(0)))
        # print(type(Q))
        # print(type(theta))
        p_hat   = evaluate_p_hat(
            H_train.unsqueeze(0), Q, theta).squeeze(0)    # [n_class, n_train_sample]
        
    # uniformly sample points in the embedding space
    # - the limits of the embedding space
    min_H        = torch.cat((H_train, H_test), 0).min(dim=0)[0].cpu().numpy()
    max_H        = torch.cat((H_train, H_test), 0).max(dim=0)[0].cpu().numpy()

    min_H, max_H = min_H - (max_H - min_H) * .1, max_H + (max_H - min_H) * .1

    n_grid = 100
    H_space      = [ np.linspace(min_h, max_h, n_grid + 1)[:-1] 
        for min_h, max_h in zip(min_H, max_H) ]           # (n_feature [n_grid])
    H            = [ [x, y] for x in H_space[0] for y in H_space[1] ]
    H            = torch.Tensor(H)                        # [n_grid * n_grid, n_feature]

    H_train = H_train.cpu()
    p_hat = p_hat.cpu()
    H_test = H_test.cpu()

    # perform test
    p_hat_knn           = knn_regressor(H, H_train, p_hat, K)    # [n_class, n_grid * n_grid]
    p_hat_kernel        = kernel_smoother(H, H_train, p_hat, h = h)  # [n_class, n_grid * n_grid]
    p_hat_simple_knn    = simple_knn_regressor(H, H_train, Y_train, K)    # [n_class, n_grid * n_grid]

    # p_hat_sft = softmax_regressor(H, H_train, p_hat)  # [n_class, n_grid * n_grid]

    H = H.to(device)

    # Set the model to evaluation mode (no gradients)
    NN_cls.eval()
    cls_p = nn.ProbLayer()

    # Make predictions on the testing data
    with torch.no_grad():
        p_hat_gcn = cls_p(NN_cls(H))
    
    p_hat_gcn = p_hat_gcn.T.cpu()
    p_hat_simple_knn = torch.tensor(p_hat_simple_knn.T)

    # print("p_hat_gcn shape", p_hat_gcn.shape)
    # print("p_hat_simple_knn shape", p_hat_simple_knn.shape)
    # print("p_hat_knn shape", p_hat_knn.shape)

    n_train_sample = H_train.shape[0]
    n_test_sample  = H_test.shape[0]

    # # print(Y_train.shape)
    # sorted_index = torch.argsort(Y_train)
    # Y_train = torch.index_select(Y_train, 0, sorted_index.squeeze())

    # sorted_index = torch.argsort(Y_test)
    # Y_test = torch.index_select(Y_test, 0, sorted_index.squeeze())

    H_train = H_train.cpu()
    p_hat = p_hat.cpu()
    H_test = H_test.cpu()
    Y_train = Y_train.cpu().numpy()
    # Y_test = Y_test.cpu()

    # print("Y_train length", len(Y_train))
    # print("Y_test length", len(Y_test))
    # print("p_hat", p_hat.shape)
    
    visualize_2Dspace_Nclass(
        n_grid, max_H, min_H, p_hat_simple_knn,
        H_train, Y_train, H_test, Y_test, classes, K, h, prefix="test")
    
    visualize_2Dspace_Nclass(
        n_grid, max_H, min_H, p_hat_knn,
        H_train, Y_train, H_test, Y_test, classes, K, h, prefix="test")
    
    visualize_2Dspace_Nclass(
        n_grid, max_H, min_H, p_hat_gcn,
        H_train, Y_train, H_test, Y_test, classes, K, h, prefix="test")
    
    visualize_2Dspace_Nclass(
        n_grid, max_H, min_H, p_hat_kernel,
        H_train, Y_train, H_test, Y_test, classes, K, h, prefix="test")