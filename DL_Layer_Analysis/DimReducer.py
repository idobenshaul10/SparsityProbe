import numpy as np
import importlib
import time
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from dataclasses import dataclass


@dataclass
class DimensionalityReducer:
    threshold_dimension: int = 4096
    output_dimension: int = 1000
    counter: int = 0

    def compute_section_number(self, dataset_size):
        return dataset_size // self.threshold_dimension
        # return 5000 // self.threshold_dimension

    def __call__(self, X):
        if self.threshold_dimension > X.shape[0]:
            return X

        # print(f"computing TruncatedSVD, X:{X.shape}, output_dimension:{self.output_dimension}")
        t0 = time.time()
        truncatedSVD = TruncatedSVD(n_components=self.output_dimension)
        _embedded = truncatedSVD.fit_transform(X)
        t1 = time.time()
        # print(f"TruncatedSVD took {t1-t0}", flush=True)
        self.counter += 1
        return _embedded
