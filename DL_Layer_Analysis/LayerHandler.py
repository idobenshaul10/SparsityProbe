import torch
from tqdm import tqdm
from DL_Layer_Analysis.DimReducer import DimensionalityReducer

class LayerHandler():
	'''Allows to run model up to layer, 
	and output different computations	
	'''
	def __init__(self, model, loader, layer, apply_dim_reduction=True):
		self.model = model
		self.model.eval()
		self.layer = layer
		self.loader = loader		
		self.layer_features = None
		self.batch_size = self.loader.batch_size
		self.use_cuda = torch.cuda.is_available()
		self.apply_dim_reduction = apply_dim_reduction
		self.dim_reducer = None
		if self.apply_dim_reduction:
			self.dim_reducer = DimensionalityReducer()

	def get_activation(self):
		def hook(model, input, output):
			print("in hook")
			if self.layer_features is None:
				self.layer_features = output.detach().view(self.batch_size, -1).cpu()
				print(output.shape, self.layer_features.shape)
			else:
				try:					
					new_outputs = output.detach().view(-1, self.layer_features.shape[1]).cpu()
					self.layer_features = torch.cat((self.layer_features, new_outputs), dim=0)
					print(output.shape, self.layer_features.shape)
				except:
					print("problems in output!")	
					pass			
		return hook	

	def __call__(self):
		handle = self.layer.register_forward_hook(self.get_activation())
		for i, (data, target) in tqdm(enumerate(self.loader), total=len(self.loader)):
			if self.use_cuda:			
				data = data.cuda()
				print("running")
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