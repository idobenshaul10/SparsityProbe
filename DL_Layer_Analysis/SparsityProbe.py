from tqdm import tqdm
import torch
from torch import nn
from DL_Layer_Analysis.LayerHandler import LayerHandler
from DL_Layer_Analysis.ModelHandler import ModelHandler
from tree_models.random_forest import WaveletsForestRegressor
from dataclasses import dataclass
import numpy as np


@dataclass
class SparsityProbe:
    loader: torch.utils.data.DataLoader
    model: torch.nn.Module
    model_handler: ModelHandler = None
    apply_dim_reduction: bool = False
    labels: torch.tensor = None
    epsilon_1: float = 0.1
    epsilon_2: float = 0.4
    mode: str = 'classification'
    n_trees: int = 5
    depth: int = 4
    n_features: str = 'auto'
    n_state: int = 2000
    norm_normalization: str = 'volume'
    text: str = ''
    output_folder: str = ''

    def __post_init__(self):
        self.model_handler = ModelHandler(self.model)
        self.labels = self.get_labels()

    def get_labels(self) -> torch.tensor:
        # picks = np.random.permutation(5000)
        return torch.tensor(self.loader.dataset.targets)#[picks]

    def aggregate_scores(self, scores) -> float:
        """at the moment we use mean aggregation for alpha scores"""
        return scores.mean()

    def get_layers(self) -> list:
        return self.model_handler.layers

    def train_tree_model(self, x, y, mode='regression',
                         trees=5, depth=9, features='auto',
                         state=2000, nnormalization='volume'):

        model = WaveletsForestRegressor(mode=mode, trees=trees, depth=depth, features=features,
                                        seed=state, norms_normalization=nnormalization)

        model.fit(x, y)
        return model

    def run_smoothness_on_features(self, features, labels=None) -> float:
        if labels is None:
            labels = self.labels
        tree_model = self.train_tree_model(features, labels,
                                           trees=self.n_trees, depth=self.depth, features=self.n_features,
                                           state=self.n_state, nnormalization=self.norm_normalization,
                                           mode=self.mode)

        alpha = tree_model.evaluate_angle_smoothness(text='try',
                                                     output_folder=self.output_folder, epsilon_1=self.epsilon_1,
                                                     epsilon_2=self.epsilon_2)

        alpha = self.aggregate_scores(alpha)
        return alpha

    def run_smoothness_on_layer(self, layer: torch.nn.Module) -> float:
        print(f"computing smoothness on:{layer.split('(')[0]}")
        # layer_handler = None
        layer_handler = LayerHandler(model=self.model, loader=self.loader,
                                     layer=layer, apply_dim_reduction=self.apply_dim_reduction)

        with layer_handler as lh:
            layer_features = lh()

        score = self.run_smoothness_on_features(layer_features)
        return score

    def compute_generalization(self) -> float:
        if self.model_handler.layers is None:
            layer = self.model_handler.get_final_layer()
        else:
            layer = self.model_handler.layers[-1]

        self.run_smoothness_on_layer(layer)
        print(f"generalization score for model is: {score}")
        return score
