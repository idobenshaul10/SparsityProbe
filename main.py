import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import importlib
from utils.utils import *
import time
import json
from collections import defaultdict
import pickle
from matplotlib.pyplot import plot, ion, show
from DL_Layer_Analysis.SparsityProbe import SparsityProbe
from torch.utils.data import Sampler
import wandb

ion()
wandb.init(project='vit_sparsity', entity='ibenshaul')


def get_args():
    parser = argparse.ArgumentParser(description='Network Smoothness Script')
    parser.add_argument('--trees', default=3, type=int, help='Number of trees in the forest.')
    parser.add_argument('--depth', default=15, type=int, help='Maximum depth of each tree.Use 0 for unlimited depth.')
    parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--env_name', type=str, default="mnist_1d_env")
    parser.add_argument('--checkpoints_folder', type=str, default=None)
    parser.add_argument('--epsilon_1', type=float, default=0.1)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--feature_dimension', default=2500, type=int, \
                        help='wanted feature dimension')
    args = parser.parse_args()
    args.epsilon_2 = 4 * args.epsilon_1
    return args


def init_params(args=None):
    if args is None:
        args = get_args()

    args.batch_size = args.batch_size
    args.use_cuda = torch.cuda.is_available()
    print(args)
    wandb.config.update(args)

    m = '.'.join(['environments', args.env_name])
    module = importlib.import_module(m)
    environment = eval(f"module.{args.env_name}()")
    # TODO: here we assert the layers used, we can let the user decide which layers
    __, train_dataset, test_dataset, __ = environment.load_enviorment()
    model = environment.get_model()
    if torch.cuda.is_available():
        model = model.cuda()

    # picks = np.random.permutation(5000)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=False)
    # sampler=picks)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args, model, train_dataset, test_dataset, train_loader


if __name__ == '__main__':
    args, model, dataset, test_dataset, data_loader = init_params()
    probe = SparsityProbe(data_loader, model, apply_dim_reduction=True, epsilon_1=args.epsilon_1,
                          epsilon_2=args.epsilon_2, n_trees=args.trees, depth=args.depth, n_state=args.seed)

    for layer in tqdm(probe.model_handler.layers):
        alpha_score, alphas = probe.run_smoothness_on_layer(layer)
        layer_name = layer._get_name()
        print(f"alpha_score for {layer_name} is {alpha_score}")
        wandb.log({"layer": layer_name, "alpha": alpha_score, "alphas_std": alphas.std()})

    # import pdb; pdb.set_trace()
