# from __future__ import print_function
import os
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
# from matplotlib.pyplot import plot, ion, show
# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
# from utils import *
import umap


def kmeans_cluster(X, Y, total_num_layers=-1, \
	visualize=False, output_folder=None, layer_str="", \
	sample_size=2500, save_graph=False, fig=None):

	plt.clf()
	plt.figure(num=None, figsize=(8, 6))
	np.random.seed(2)
	k = len(np.unique(Y))
	print(f"Fitting k means with k={k}")
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X)	
	return kmeans

def create_umap(X, Y, total_num_layers, output_folder, \
	layer_str, sample_size=2500, save_graph=False, fig=None):

	print(f"Fitting umap")
	reducer = umap.UMAP(random_state=42)
	indices = np.random.choice(len(X), sample_size)

	X = X[indices]
	Y = Y[indices]	
	
	embedding_train = reducer.fit_transform(X)
	print(f"Done fitting umap")
	
	if fig is None:
		fig = plt.figure(figsize=(20, 16))

	print("layers:", total_num_layers//2, (2+total_num_layers%2), int(layer_str)+2)
	ax = fig.add_subplot(total_num_layers//2, (2+total_num_layers%2), int(layer_str)+2)
	scatter = ax.scatter(
		embedding_train[:, 0], embedding_train[:, 1], c=Y, cmap="Spectral" , s=0.5
	)
	legend1 = ax.legend(*scatter.legend_elements(),
			loc="lower left", title="Classes")

	plt.setp(ax, xticks=[], yticks=[])	
	plt.suptitle(f"UMAP of Clustering for {layer_str}", fontsize=18)
	ax.set_xlabel(f"layer:{layer_str}")

	if save_graph and output_folder is not None:
		save_path = os.path.join(output_folder, f"{layer_str}.png")
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')

def get_clustering_statistics(X, Y, kmeans):
	Y = Y.squeeze()	
	metrics_results = {}
	preds = kmeans.labels_
	print("START clustering statistics")	
	metrics_results['adj_rand'] = metrics.adjusted_rand_score(Y, preds)
	metrics_results['MI_score'] = metrics.adjusted_mutual_info_score(Y, preds)
	metrics_results['homogeneity_score'] = metrics.homogeneity_score(Y, preds)
	metrics_results['completeness'] = metrics.completeness_score(Y, preds)
	metrics_results['FMI'] = metrics.fowlkes_mallows_score(Y, preds)
	print("DONE clustering statistics")
	for k, v in metrics_results.items():
		print(f'{k}:{v}')
	return metrics_results