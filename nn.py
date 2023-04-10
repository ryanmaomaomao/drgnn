#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script defines multiple neural networks for the embedding purpose.
References:
- https://github.com/pytorch/examples/blob/master/dcgan/main.py#L157
- 
"""
# decouple the classfication layer and gcn model layer
# for better programming style

# GCN + NN classifier + softmax
# or 
# GCN + softmax


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim = 16, output_dim = 7, num_layers = 2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.conv2 = GCNConv(hidden_dim, output_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p = 0.5)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training, p = 0.5)
        x = self.conv2(x, edge_index)

        return x

        # return F.log_softmax(x, dim=1)

class NN_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, linear_hidden_size = 16, keepprob = 0.5):
        super().__init__()
        self.keepprob = keepprob
        self.fc1   = torch.nn.Linear(input_dim, linear_hidden_size)
        self.fc2   = torch.nn.Linear(linear_hidden_size, output_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc3   = torch.nn.Linear(linear_hidden_size, output_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p = (1-self.keepprob), training=self.training)
        x = F.relu(x)
        output = self.fc2(x)
        # output = self.fc3(x)

        return output
        
class ClassficationLayer(torch.nn.Module):
    '''
    log softmax layer for 
    '''
    def __init__(self):
        super().__init__()
        # self.keepprob = keepprob
        # self.conv1 = GCNConv(in_feature, hidden_size) # in_channels, out_channels
        # # self.conv2 = GCNConv(16, dataset.num_classes)
        # self.conv2 = GCNConv(hidden_size, out_feature) 
        # self.pretraining = pretraining

    def forward(self, x):
        return F.log_softmax(x, dim=1)

class ProbLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.keepprob = keepprob
        # self.conv1 = GCNConv(in_feature, hidden_size) # in_channels, out_channels
        # # self.conv2 = GCNConv(16, dataset.num_classes)
        # self.conv2 = GCNConv(hidden_size, out_feature) 
        # self.pretraining = pretraining

    def forward(self, x):
        return F.softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, heads, dropout = 0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        # self.lin = torch.nn.Linear(hidden_dim * heads, num_classes)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1,concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.conv1.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.dropout(x, p=conv.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        # x = x.view(-1, self.lin.in_features)
        # x = self.lin(x)
        x = F.dropout(x, p=self.conv2.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        # return F.log_softmax(x, dim=-1)

        