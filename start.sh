#!/bin/bash

# gnn low data
python3 main.py \
    --dataset Cora \
    --seed 0 \
    --sample_per_class 5 \
    --n_class 7 \
    --n_feature 7 \
    --batch_class_size 4 \
    --max_theta 0.01 \
    --lr 0.01 \
    --K 14 \
    --h 0.001 \
    --gnn_type gcn\
    --gnn_hidden_dim 7 \
    --gnn_hidden_layers 2 \
    --gnn_early_stopping False \
    --dr_epochs 501 \
    --dr_early_stopping True \
    --edge_removel_rate 0 \
    --noise_rate 0 \
    --exp_name test_main \

# # gnn low data
# python3 main.py \
#     --dataset Cora \
#     --seed 0 \
#     --sample_per_class 5 \
#     --n_class 7 \
#     --n_feature 7 \
#     --batch_class_size 4 \
#     --max_theta 0.01 \
#     --lr 0.01 \
#     --K 14 \
#     --h 0.001 \
#     --gnn_type gcn\
#     --gnn_hidden_dim 7 \
#     --gnn_hidden_layers 2 \
#     --gnn_early_stopping False \
#     --dr_epochs 501 \
#     --dr_early_stopping True \
#     --edge_removel_rate 0 \
#     --noise_rate 0 \
#     --exp_name test_main \

# # gnn edge removal
# python3 main.py \
#     --dataset Cora \
#     --seed 0 \
#     --sample_per_class 5 \
#     --n_class 7 \
#     --n_feature 7 \
#     --batch_class_size 4 \
#     --max_theta 0.01 \
#     --lr 0.01 \
#     --K 14 \
#     --h 0.001 \
#     --gnn_type gcn\
#     --gnn_hidden_dim 7 \
#     --gnn_hidden_layers 2 \
#     --gnn_early_stopping False \
#     --dr_epochs 501 \
#     --dr_early_stopping True \
#     --edge_removel_rate 0 \
#     --noise_rate 1 \
#     --exp_name test_main \