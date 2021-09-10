import numpy as np
import importlib
import time
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from dataclasses import dataclass
import wandb

@dataclass
class DimensionalityReducer:
    threshold_dimension: int = 4096
    output_dimension: int = 1000
    counter: int = 0
    refit_every_batch: bool = True
    truncatedSVD: TruncatedSVD = None
    DR_fitted = False

    def __post_init__(self):
        self.truncatedSVD = TruncatedSVD(n_components=self.output_dimension)
        wandb.config.update({"dim_reducer_threshold_dimension": self.threshold_dimension})
        wandb.config.update({"dim_reducer_output_dimension": self.output_dimension})
        wandb.config.update({"refit_every_batch": self.refit_every_batch})

    def compute_section_number(self, dataset_size):
        return dataset_size // self.threshold_dimension
        # return 5000 // self.threshold_dimension

    def __call__(self, X):
        if self.threshold_dimension > X.shape[0]:
            return X

        if self.refit_every_batch:
            if not self.DR_fitted:
                self.truncatedSVD.fit(X)
                self.DR_fitted = True
            t0 = time.time()
            _embedded = self.truncatedSVD.transform(X)
            t1 = time.time()
        else:
            _embedded = self.truncatedSVD.fit_transform(X)

        self.counter += 1
        return _embedded
