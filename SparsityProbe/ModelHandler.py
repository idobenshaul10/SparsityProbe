import torch
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass
from SparsityProbe.LayerHandler import LayerHandler
from SparsityProbe.DimReducer import DimensionalityReducer

@dataclass
class ModelHandler:
    """Analyzes the model to find intermediate layers"""
    model: torch.nn.Module
    layers: list = None

    def __post_init__(self):
        if self.layers is None:
            self.init_layers()

    def init_layers(self):
        '''
        returns all layers of the model that have parameters
        and are not BatchNorm layers. This is not exaustive
        and should be improved
        '''
        try:
            layers = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight'):
                    layers.append(module)
            self.layers = layers
        except Exception as e:
            print(f"problems in fetching model layers:{e}")
            self.layers = []

    def __call__(self):
        return self.layers

    def get_final_layer(self):
        return self.layers[-1]


if __name__ == '__main__':
    from torchvision.models import resnet18
    model = resnet18()
    modelHandler = ModelHandler(model)
    final_layer = modelHandler.get_final_layer()
