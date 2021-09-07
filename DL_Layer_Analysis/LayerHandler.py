import torch
from tqdm import tqdm
from DimReducer import DimensionalityReducer
from dataclasses import dataclass
import numpy as np

@dataclass
class LayerHandler:
	'''Allows to run model up to layer, 
	and output different computations	
	'''
	model: torch.nn.Module
	loader: torch.utils.data.DataLoader
	layer: torch.nn.Module
	apply_dim_reduction: bool
	layer_features: np.array = None
	dim_reducer: DimensionalityReducer = None
	batch_size: int = None
	use_cuda: bool = None

	def __post_init__(self):
		self.model.eval()
		self.batch_size = self.loader.batch_size
		self.use_cuda = torch.cuda.is_available()
		if self.apply_dim_reduction:
			self.dim_reducer = DimensionalityReducer()

	def get_activation(self):
		def hook(model, input, output):
			if self.layer_features is None:
				self.layer_features = output.detach().view(self.batch_size, -1).cpu()
			else:
				try:					
					new_outputs = output.detach().view(-1, self.layer_features.shape[1]).cpu()
					self.layer_features = torch.cat((self.layer_features, new_outputs), dim=0)
				except:
					print("problems in output!")	
					pass
		return hook	

	def __call__(self):
		handle = self.layer.register_forward_hook(self.get_activation())
		for i, (data, target) in tqdm(enumerate(self.loader), total=len(self.loader)):
			if self.use_cuda:			
				data = data.cuda()
			self.model(data)
			del data

		layer_features = self.layer_features.numpy()			
		handle.remove()
		if self.apply_dim_reduction:
			layer_features = self.dim_reducer(layer_features)
		return layer_features

	def dispose(self):
		# !!TODO!!
		pass