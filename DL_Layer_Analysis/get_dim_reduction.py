import numpy as np
import importlib
import time
import umap
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

np.random.seed(2)
def get_dim_reduction_UMAP(X, output_dimension=1000):
	print(f"computing umap")	
	t0 = time.time()
	reducer = umap.UMAP(random_state=42, n_components=output_dimension)	
	t1 = time.time()	
	embedding_train = reducer.fit_transform(X)
	print(f"umap took {t1-t0}", flush=True)	
	return embedding_train

def get_dim_reduction_PCA(X, output_dimension=1000):
	print(f"computing PCA")
	t0 = time.time()

	pca = PCA(n_components=output_dimension)

	_embedded = pca.fit_transform(X)
	t1 = time.time()
	print(f"PCA took {t1-t0}", flush=True)	
	return _embedded

def get_dim_reduction(X, output_dimension=1000):
	print(f"computing TruncatedSVD, X:{X.shape}, output_dimension:{output_dimension}")
	t0 = time.time()

	truncatedSVD = TruncatedSVD(n_components=output_dimension)
	_embedded = truncatedSVD.fit_transform(X)

	t1 = time.time()
	print(f"TruncatedSVD took {t1-t0}", flush=True)	
	return _embedded




