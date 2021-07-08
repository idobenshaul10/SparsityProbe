import torch
from torch import nn
from tqdm import tqdm
from DL_Layer_Analysis.DimReducer import DimensionalityReducer


# class LAYER_TYPE():    
#     DENSE = auto()
#     CONV = auto()    
#     FLATTENED = auto()
#     EMBEDDING = auto()
#     NORM = auto()
LayerTypes = {'Dense': nn.Linear, }

class ModelHandler():
	'''Analyzes the model to find intermediate layers
	'''
	def __init__(self, model):
		self.model = model
		self.layers = self.init_layers()

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
				if hasattr(module, 'weight'):
					layers.append(module)				
			return layers
		except Exception as e:
			# print(f"problems in fetching model layers:{e}")
			return []

	def __call__(self):
		return self.layers

	def get_final_layer(self):
		return self.layers[-1]

if __name__ == '__main__':
	from torchvision.models import resnet18
	model = resnet18()
	modelHandler = ModelHandler(model)
	modelHandler()
	print(modelHandler.get_final_layer())

