import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import importlib

currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.utils import *
import time
import json
from collections import defaultdict
import pickle
# from DL_Layer_Analysis.clustering import kmeans_cluster, \
# 	get_clustering_statistics, create_umap
# from DL_Layer_Analysis.get_dim_reduction import get_dim_reduction

from matplotlib.pyplot import plot, ion, show
from SparsityProbe import *

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
    parser.add_argument('--only_umap', action='store_true', default=False)
    parser.add_argument('--use_clustering', action='store_true', default=False)
    parser.add_argument('--calc_test', action='store_true', default=False)
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
    dict_input = vars(args)
    environment = eval(f"module.{args.env_name}()")
    __, dataset, test_dataset, __ = environment.load_enviorment()
    model = environment.get_model()

    if torch.cuda.is_available():
        model = model.cuda()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args, model, dataset, test_dataset, data_loader


activation = {}


def get_activation(name, args):
    def hook(model, input, output):
        if name not in activation:
            activation[name] = output.detach().view(args.batch_size, -1)
        else:
            try:
                new_outputs = output.detach().view(-1, activation[name].shape[1])
                activation[name] = \
                    torch.cat((activation[name], new_outputs), dim=0)
            except:
                pass

    return hook


def save_alphas_plot(args, alphas, sizes, test_stats=None, \
                     clustering_stats=None, save_to_file=False):
    plt.figure(1)
    plt.clf()
    if type(alphas) == list:
        plt.fill_between(sizes, [k[0] for k in alphas], [k[-1] for k in alphas], \
                         alpha=0.2, facecolor='#089FFF', \
                         linewidth=4)
        plt.plot(sizes, [np.array(k).mean() for k in alphas], 'k', color='#1B2ACC')
    else:
        plt.plot(sizes, alphas, 'k', color='#1B2ACC')

    plt.title(f"{args.env_name} Angle Smoothness")

    acc_txt = ''
    if test_stats is not None and 'top_1_accuracy' in test_stats:
        acc_txt = f"TEST Top1-ACC {test_stats['top_1_accuracy']}"

    plt.xlabel(f'Layer\n\n{acc_txt}')
    plt.ylabel(f'evaluate_smoothnes index- alpha')

    if save_to_file and args.output_folder is not None:
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)
        save_path = os.path.join(args.output_folder, "result.png")
        print(f"save_path:{save_path}")
        plt.savefig(save_path, \
                    dpi=300, bbox_inches='tight')

    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError

    json_file_name = os.path.join("result.json")
    write_data = {}
    write_data['alphas'] = [tuple(k) for k in alphas]
    write_data['sizes'] = sizes
    write_data['test_stats'] = test_stats
    write_data['clustering_stats'] = clustering_stats

    if save_to_file and args.output_folder is not None:
        norms_path = os.path.join(args.output_folder, json_file_name)
        with open(norms_path, "w+") as f:
            json.dump(write_data, f, default=convert)
    return write_data


def get_top_1_accuracy(model, data_loader, device):
    softmax = nn.Softmax(dim=1)
    correct_pred = 0
    n = 0
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            output = model(X)
            probs = softmax(output).cpu()
            predicted_labels = torch.max(probs, 1)[1]
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
    return (correct_pred.float() / n).item()


if __name__ == '__main__':
    args, model, dataset, test_dataset, data_loader = init_params()

    model_handler = ModelHandler(model)
    layers = model_handler.layers
    import pdb; pdb.set_trace()
    # layerHandler = LayerHandler(model, data_loader, layers[-1])
    # layer_outputs = layerHandler.run_model_up_to_layer()

    # probe = SparsityProbe(data_loader, model, None)
    probe = SparsityProbe(data_loader, model, None)
    probe.compute_generalization()
