#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader
"""

import numpy as np
import random
import torch
import itertools


def get_train_mask(number_of_sample_per_class, data, selected_class=[0, 1, 2, 3, 4, 5, 6]):
    """
    Randomly select samples from each class and get corresponding samples and labels
    :param number_of_sample_per_class: number of samples to be selected from each class
    :param data: data object
    :param selected_class: list of classes to be selected from
    :return: result of samples
    """
    flags = data.train_mask.detach().cpu().numpy()
    labels = data.y.detach().cpu().numpy()

    assert (len(flags) == len(labels)), "train mask and Y should be equal in length"

    # Randomly select samples from each class
    sample_indices = []
    # iterate over the classes
    for j in selected_class:
        # select sample if it's a training sample and it belongs to the class
        indices = [index for index, label in enumerate(labels) if label == j and flags[index]]
        # randomly sample the indices
        selected_indices = random.sample(indices, number_of_sample_per_class)
        sample_indices.extend(selected_indices)

    # get the corresponding samples and labels
    result = [True if x in sample_indices else False for x in range(len(data.train_mask))]

    return result


# def get_train_combinations(number_of_sample_per_class, data, selected_class=[0, 1, 2, 3, 4, 5, 6]):
#     """
#     Generate a list of all combinations of selecting number_of_sample_per_class samples from each selected class in a dataset
#     :param number_of_sample_per_class: number of samples to be selected from each class
#     :param data: data object
#     :param selected_class: list of classes to be selected from
#     :return: a list of all combinations of selecting number_of_sample_per_class samples from each selected class in a dataset
#     """
#     flags = data.train_mask.detach().cpu().numpy()
#     labels = data.y.detach().cpu().numpy()

#     assert (len(flags) == len(labels)), "train mask and Y should be equal in length"

#     sample_indices = []
#     # iterate over the selected classes and select samples from each class
#     for j in selected_class:
#         indices = [index for index, label in enumerate(labels) if label == j and flags[index]]
#         selected_indices = random.sample(indices, number_of_sample_per_class)
#         sample_indices.append(selected_indices)

#     # generate all combinations of the selected indices
#     combinations = list(itertools.product(*sample_indices))

#     # create a list of boolean masks for each combination
#     result = []
#     for c in combinations:
#         mask = [False] * len(flags)
#         for index in c:
#             mask[index] = True
#         result.append(mask)

#     return result

def add_noise_to_data(data, pct):
    """
    Add noise to the dataset
    :param dataset: dataset to be added noise
    :param sigma: noise strength
    :param pct: percentage of the dataset to add noise
    :return: data with added noise
    """
    # data = dataset[0]
    sigma = torch.std(data.x).item()

    dense_array = np.array(data.x)

    # Convert the Numpy array to a PyTorch tensor
    atensor = torch.from_numpy(dense_array)

    # add noise
    atensor += torch.randn(atensor.shape[0], atensor.shape[1]) * sigma * pct
    data.x = atensor

    return data

def remove_edge_to_data(data, drop_rate):
    """
    Remove edges from the dataset
    :param dataset: dataset to be removed edges
    :param drop_rate: the percentage of edges to be removed
    :return: data with removed edges
    """
    # data = dataset[0]

    idx = torch.randperm(data.edge_index.size(1))[:int((1 - drop_rate) * data.edge_index.size(1))]
    data.edge_index = data.edge_index[:, idx]

    return data