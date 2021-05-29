# from __future__ import print_function
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
# from torch.utils.data import TensorDataset, DataLoader
import pickle
from DL_Layer_Analysis.clustering import kmeans_cluster, \
	get_clustering_statistics, create_umap
from DL_Layer_Analysis.get_dim_reduction import get_dim_reduction
from matplotlib.pyplot import plot, ion, show
#  python .\DL_smoothness.py --env_name cifar10_env --use_clustering

ion()

def get_args():	
	parser = argparse.ArgumentParser(description='Network Smoothness Script')	
	parser.add_argument('--trees',default=1,type=int,help='Number of trees in the forest.')	
	parser.add_argument('--depth', default=15, type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')	
	# parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the regressor.')
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
	args.epsilon_2 = 4*args.epsilon_1
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
	# params_path = os.path.join(args.checkpoints_folder, 'args.p')
	# if os.path.isfile(params_path):
	# 	params = vars(pickle.load(open(params_path, 'rb')))

	__, dataset, test_dataset, __ = environment.load_enviorment()	
	model = environment.get_model()
	# checkpoint_path = os.path.join(args.checkpoints_folder, args.checkpoint_file_name)
	# model = environment.get_model(**params)
	# checkpoint = torch.load(checkpoint_path)['checkpoint']	
	# model.load_state_dict(checkpoint)	
	if torch.cuda.is_available():
		model = model.cuda()
	
	layers = environment.get_layers(model)

	data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	# addition = f"{args.env_name}_{args.trees}_{args.depth}_{args.high_range_epsilon}_{args.low_range_epsilon:.2f}_{args.seed}"

	# args.output_folder = os.path.join(args.output_folder, addition)
	# if not os.path.isdir(args.output_folder):
	# 	os.mkdir(args.output_folder)

	return args, model, dataset, test_dataset, layers, data_loader

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
		plt.plot(sizes, [np.array(k).mean()	 for k in alphas], 'k', color='#1B2ACC')
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


def run_smoothness_analysis(args, model, dataset, test_dataset, layers, data_loader):	
	Y = torch.cat([target for (data, target) in tqdm(data_loader)]).detach()	
	norm_normalization = 'num_samples'	
	visualize_umap = True

	model.eval()
	sizes, alphas = [], []
	clustering_stats = defaultdict(list)
	layers_to_run_on = [-1] + list(range(len(layers)))	
	total_num_layers = len(layers_to_run_on)
	print(f"There are {total_num_layers} layers!")
	fig = plt.figure(figsize=(20, 16))

	with torch.no_grad():		
		for k in layers_to_run_on:
			layer_str = 'layer'
			print(f"LAYER {k}, type:{layer_str}")
			layer_name = f'layer_{k}'
			if k == -1:
				X = torch.cat([data for (data, target) in tqdm(data_loader)]).detach()
				X = X.view(X.shape[0], -1)

				if X.shape[1] > args.feature_dimension:
					X = get_dim_reduction(X, args.feature_dimension)

			else:
				result = None				
				handle = layers[k].register_forward_hook(get_activation(layer_name, args))					
				for i, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):	
					if args.use_cuda:
						data = data.cuda()
					model(data)
					del data

				cur_X = activation[list(activation.keys())[0]]
				if result is None:
					result = cur_X.cpu().numpy()
				else:
					result += cur_X.cpu().numpy()					
				handle.remove()
				del cur_X
				del activation[layer_name]
				
				X = result
				if X.shape[1] > args.feature_dimension:
					X = get_dim_reduction(X, args.feature_dimension)
					print("after dim reduction")
			
			start = time.time()
			Y = np.array(Y).reshape(-1, 1)			

			print(f"X.shape:{X.shape}, Y shape:{Y.shape}")
			assert(Y.shape[0] == X.shape[0])			
			
			if not args.only_umap:
				try:
					alpha_index = run_alpha_smoothness(X, Y, t_method="WF", \
						n_trees=args.trees, \
						m_depth=args.depth, \
						n_state=args.seed, norm_normalization=norm_normalization, 
						text=f"layer_{k}_{layer_str}", output_folder=args.output_folder, 
						epsilon_1=args.epsilon_1, epsilon_2=args.epsilon_2)					
					
					print(f"ALPHA for LAYER {k} is {np.mean(alpha_index)}")
					if args.use_clustering:
						kmeans = kmeans_cluster(X, Y, total_num_layers=total_num_layers, \
							visualize=False, output_folder=args.output_folder, \
							layer_str=f"{k}", fig=fig)
						clustering_stats[k] = get_clustering_statistics(X, Y, kmeans)
						if visualize_umap:
							create_umap(X, Y, total_num_layers=total_num_layers, \
								output_folder=args.output_folder, \
								layer_str=f"{k}", fig=fig)

					sizes.append(k)
					alphas.append(alpha_index)
				except Exception as error:
					print(f"error: {error}, skipping layer:{k}")

			else:					
				create_umap(X, Y, total_num_layers=total_num_layers, \
					output_folder=args.output_folder, \
					layer_str=f"{k}", fig=fig, save_graph=True)
	plt.show()

	if not args.only_umap:
		test_stats = None
		if args.calc_test and test_dataset is not None:
			test_stats = {}			
			test_loader = torch.utils.data.DataLoader(test_dataset, \
				batch_size=args.batch_size, shuffle=False)
			device = 'cuda' if args.use_cuda else 'cpu'
			test_accuracy = get_top_1_accuracy(model, test_loader, device)
			test_stats['top_1_accuracy'] = np.mean(test_accuracy)
		return save_alphas_plot(args, alphas, sizes, \
			test_stats, clustering_stats, save_to_file=True)

if __name__ == '__main__':
	args, model, dataset, test_dataset, layers, data_loader = init_params()
	run_smoothness_analysis(args, model, dataset, test_dataset, layers, data_loader)

