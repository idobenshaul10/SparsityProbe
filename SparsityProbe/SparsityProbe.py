import os.path

from tqdm import tqdm
import torch
from torch import nn
from SparsityProbe.LayerHandler import LayerHandler
from SparsityProbe.ModelHandler import ModelHandler
from tree_models.random_forest import WaveletsForestRegressor
from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class SparsityProbe:
    loader: torch.utils.data.DataLoader
    model: torch.nn.Module
    model_handler: ModelHandler = None
    apply_dim_reduction: bool = False
    labels: torch.tensor = None
    compute_using_index: bool = False
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
    layers: list = None
    layer_feature_cache_folder: str = None

    def __post_init__(self):
        self.model_handler = ModelHandler(self.model, self.layers)
        self.labels = self.get_labels()

    def get_labels(self) -> torch.tensor:
        try:
            return torch.tensor(self.loader.dataset.targets)  # [picks]
        except:
            try:
                return torch.tensor([k[1].item() for k in self.loader.dataset])
            except:
                return torch.tensor(self.loader.dataset['labels'])

    def aggregate_scores(self, scores) -> float:
        """at the moment we use mean aggregation for alpha scores"""
        return scores.mean()

    def get_layers(self) -> list:
        return self.model_handler.layers

    def train_tree_model(self, x, y, mode='classification',
                         trees=5, depth=9, features='auto',
                         state=2000, nnormalization='volume') -> WaveletsForestRegressor:

        model = WaveletsForestRegressor(mode=mode, trees=trees, depth=depth,
                                        seed=state, norms_normalization=nnormalization)

        model.fit(x, y)
        return model

    def run_smoothness_on_features(self, features, labels=None) -> tuple:
        if labels is None:
            labels = self.labels
        tree_model = self.train_tree_model(features, labels,
                                           trees=self.n_trees, depth=self.depth, features=self.n_features,
                                           state=self.n_state, nnormalization=self.norm_normalization,
                                           mode=self.mode)
        print("evaluating smoothness")
        alphas = tree_model.evaluate_angle_smoothness(output_folder=self.output_folder, epsilon_1=self.epsilon_1,
                                                      epsilon_2=self.epsilon_2, compute_using_index=self.compute_using_index)

        aggregated_alpha = self.aggregate_scores(alphas)
        return aggregated_alpha, alphas

    def run_smoothness_on_layer(self, layer: torch.nn.Module, text: str='') -> tuple:
        print(f"\ncomputing smoothness on:{layer._get_name()}")
        layer_handler = LayerHandler(model=self.model, loader=self.loader,
                                     layer=layer, apply_dim_reduction=self.apply_dim_reduction)


        if self.layer_feature_cache_folder is not None:
            Path(self.layer_feature_cache_folder).mkdir(exist_ok=True, parents=True)
            output_path = os.path.join(self.layer_feature_cache_folder, f"{text}.npz")
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            if os.path.isfile(output_path):
                # del self.loader
                print(f"loading layer features from {output_path}")
                layer_features = np.load(output_path)['layer_features']
            else:
                print(f"no cache found at {output_path}, computing layer features")
                with layer_handler as lh:
                    layer_features = lh()
                    np.savez(output_path, layer_features=layer_features)
                    print(f"saved to {output_path}")
        else:
            with layer_handler as lh:
                layer_features = lh()

        print(f"layer_features:{layer_features.shape}")

        mean_alpha, alphas = self.run_smoothness_on_features(layer_features)
        return mean_alpha, alphas

    def compute_generalization(self) -> tuple:
        if self.model_handler.layers is None:
            layer = self.model_handler.get_final_layer()
        else:
            layer = self.model_handler.layers[-2]

        mean_alpha, alphas = self.run_smoothness_on_layer(layer)
        print(f"generalization score for model is: {mean_alpha}")
        return mean_alpha, alphas
