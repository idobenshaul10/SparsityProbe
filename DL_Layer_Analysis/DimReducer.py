import numpy as np
import importlib
import time
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

class DimensionalityReducer:
	def __init__(self, threshold_dimension=2048*2, output_dimension=1000):
		if threshold_dimension < output_dimension:
			print(f"this reduction seems futile, TH:{threshold_dimension}," +\
				f" output:{output_dimension}")
		
		self.threshold_dimension = threshold_dimension
		self.output_dimension = output_dimension
		self.counter = 0

	def compute_section_number(self, dataset_size):
		return dataset_size//self.threshold_dimension


	def __call__(self, X):
		if self.threshold_dimension > X.shape[0]:
			return X

		print(f"computing TruncatedSVD, X:{X.shape}, output_dimension:{self.output_dimension}")
		t0 = time.time()
		truncatedSVD = TruncatedSVD(n_components=self.output_dimension)
		_embedded = truncatedSVD.fit_transform(X)
		t1 = time.time()
		print(f"TruncatedSVD took {t1-t0}", flush=True)
		self.counter += 1
		return _embedded





