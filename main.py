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
# from SparsityProbe import *
ion()


def get_args():
    parser = argparse.ArgumentParser(description='Network Smoothness Script')
    parser.add_argument('--trees', default=1, type=int, help='Number of trees in the forest.')
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

    m = '.'.join(['environments', args.env_name])
    module = importlib.import_module(m)
    environment = eval(f"module.{args.env_name}()")
    # TODO: here we assert the layers used, we can let the user decide which layers
    __, train_dataset, test_dataset, __ = environment.load_enviorment()
    model = environment.get_model()
    if torch.cuda.is_available():
        model = model.cuda()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args, model, train_dataset, test_dataset, train_loader


if __name__ == '__main__':
    args, model, dataset, test_dataset, data_loader = init_params()
    probe = SparsityProbe(data_loader, model, apply_dim_reduction=False)
    layer = probe.model_handler.layers[-2]

    alpha_score = probe.run_smoothness_on_layer(layer)
    print(f"alpha_score for {layer} is {alpha_score}")
    # probe.compute_generalization()
