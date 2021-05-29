import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from itertools import cycle
import seaborn as sns
import matplotlib.gridspec as gridspec
import copy
#USAGE: python .\DL_Layer_Analysis\plot_DL_json_results.py --main_dir RFWFC\results\mnist

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')	
	parser.add_argument('--main_dir', type=str ,help='Results folder', default=None)
	parser.add_argument('--checkpoints','--list', nargs='+', default=None)	
	args = parser.parse_args()
	return args

def plot_layers(main_dir, checkpoints=None, plot_test=True, add_fill=False, remove_layers=0, \
	use_clustering=False, remove_begin=0):
	if plot_test:
		if not use_clustering:
			fig, axes = plt.subplots(1, 2)
			gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1])
			axes = [None, None]
		else:
			fig, axes = plt.subplots(1, 3, figsize=(12, 10))
			gs = gridspec.GridSpec(1, 3, width_ratios=[10, 1, 10])
			axes = [None, None, None]	
			axes[2] = plt.subplot(gs[2])
		
		axes[0] = plt.subplot(gs[0])		
		axes[1] = plt.subplot(gs[1])
		
		
		axes[0].set_ylabel(r'$\alpha$'+'-score')
		axes[1].set_xticks([])
		axes[0].set_xticks([-1, 0, 1, 2, 3, 4, 5, 6] )		
		axes[1].set_ylabel('Test Accuracy')

		axes[0].set_title(f"Comparing " + r'$\alpha$' + "-scores")
		axes[0].set_xlabel("Layers")
		axes[1].set_title("Test Accuracy Scores")		

		if use_clustering:
			axes[2].set_title("Clustering Statistics")
			axes[2].set_xlabel("Layers")
			axes[2].set_title("Clustering Metrics")
	else:
		fig, axes = plt.subplots(1, 1)
		axes = [axes]

	if checkpoints is not None:
		file_paths = checkpoints
	else:
		file_paths = list(Path(main_dir).glob('**/*.json'))
		file_paths = [str(k) for k in file_paths]		
		try:
			file_paths.sort(key=lambda x: str(x).split('/')[-5])
		except Exception as e:
			print(f"error:{e}")
		
	clustering_stats = None
	colors = sns.color_palette("pastel", 20)
	

	test_results = []
	handles = []
	width = 0.25	

	for idx, file_path in enumerate(file_paths):
		file_path = str(file_path)
		epoch = str(file_path).split('/')[-2]		
		
		with open(file_path, "r+") as f:			
			result = json.load(f)
		
		sizes = result["sizes"]
		alphas = result["alphas"]

		if remove_layers > 0:
			sizes, alphas = sizes[:-remove_layers], alphas[:-remove_layers]
		if remove_begin > 0:
			sizes, alphas = sizes[remove_begin:], alphas[remove_begin:]

		test_stats = None
		if 'test_stats' in result:
			test_stats = result['test_stats']
		if 'clustering_stats' in result:
			clustering_stats = result['clustering_stats']
		if add_fill:
			axes[0].fill_between(sizes, [k[0] for k in alphas], [k[-1] for k in alphas], \
				alpha=0.2, linewidth=4)
		
		values = [np.array(k).mean()	 for k in alphas]
		print(values)		
		axes[0].plot(sizes, values, label=epoch, color=colors[idx%len(colors)])

		if test_stats is not None and plot_test:
			test_results.append([test_stats['top_1_accuracy']])
			axes[1].bar(idx*width, [test_stats['top_1_accuracy']], width, \
				label=str(epoch), color=colors[idx%len(colors)])			

		lines = ["-","--","-.",":", "-*", "-+"]
		linecycler = cycle(lines)

		if use_clustering:
			if clustering_stats is not None and plot_test:
				keys = sorted(list(clustering_stats.keys()))			
				if len(keys) == 0:
					continue
				stat_names = clustering_stats[list(keys)[0]].keys()			
				for chosen_stat in stat_names:				
					values = [clustering_stats[k][chosen_stat] for k in keys]				
					if idx == 0:
						h, = axes[2].plot(keys, values, next(linecycler), color=colors[idx], label=f"{chosen_stat}")	
						handles.append(copy.copy(h))
					else:
						axes[2].plot(keys, values, next(linecycler), color=colors[idx])

	
	axes[0].legend()
	for h in handles:
		h.set_color("black")
	if use_clustering:
		axes[2].legend(handles=handles)
	plt.show()

if __name__ == '__main__':
	args = get_args()	
	plot_epochs(args.main_dir, args.checkpoints, plot_test=True, add_fill=True)