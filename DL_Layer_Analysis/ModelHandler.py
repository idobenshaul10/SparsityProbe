import torch
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass
from DL_Layer_Analysis.LayerHandler import LayerHandler
from DL_Layer_Analysis.DimReducer import DimensionalityReducer


@dataclass
class ModelHandler:
    """Analyzes the model to find intermediate layers"""
    model: torch.nn.Module
    layers: list = None

    def __post_init__(self):
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
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    continue
                import pdb; pdb.set_trace()
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
# probe = SparsityProbe(data_loader, model, None)


# ViTEmbeddings,PatchEmbeddings,
# vit.encoder
# vit.encoder.layer.0, vit.encoder.layer.1
