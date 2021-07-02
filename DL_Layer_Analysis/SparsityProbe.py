from tqdm import tqdm
import torch
from torch import nn
from DL_Layer_Analysis.DimReducer import DimensionalityReducer
from tree_models.random_forest import WaveletsForestRegressor

class LayerHandler():
	'''Allows to run model up to layer, 
	and output different computations	
	'''
	def __init__(self, model, loader, layer, apply_dim_reduction=True):
		self.model = model
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
			if self.layer_features is None:
				self.layer_features = output.detach().view(self.batch_size, -1)
			else:
				try:					
					new_outputs = output.detach().view(-1, self.layer_features.shape[1])
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
		
		layer_features = self.layer_features.cpu().numpy()			
		handle.remove()
		if self.apply_dim_reduction:
			layer_features = self.dim_reducer(layer_features)

		return layer_features

	def dispose(self):
		# !!TODO!!
		pass

class SparsityProbe():
	def __init__(self, loader, model, model_layers=None, \
		apply_dim_reduction=False, epsilon_1=0.1, epsilon_2=0.4):
		self.loader = loader
		self.model = model
		self.model_layers = model_layers
		self.apply_dim_reduction = apply_dim_reduction
		self.labels = self.get_labels()
		
		# tree parameters
		self.t_method='RF'
		self.n_trees=1
		self.depth=9
		self.n_features='auto'
		self.n_state=2000
		self.norm_normalization='volume'
		self.text=''
		self.output_folder=''
		self.epsilon_1 = epsilon_1
		self.epsilon_2 = epsilon_2

	def get_labels(self):
		Y = torch.cat([target for (data, target) in tqdm(self.loader)]).detach()
		return Y

	def find_final_model_layer(self):
		pass


	def train_tree_model(self, x, y, mode='regression', trees=5, depth=9, features='auto',
				state=2000, nnormalization='volume'):

		model = WaveletsForestRegressor(mode=mode, trees=trees, depth=depth, features=features, \
			seed=state, norms_normalization=nnormalization)

		model.fit(x, y)
		return model
	
	def run_smoothness_on_features(self, features):		
		tree_model = self.train_tree_model(features, self.labels, \
			trees=self.n_trees, depth=self.depth, features=self.n_features, \
			state=self.n_state, nnormalization=self.norm_normalization)

		alpha = tree_model.evaluate_angle_smoothness(text='try', \
			output_folder=self.output_folder, epsilon_1=self.epsilon_1, \
			epsilon_2=self.epsilon_2)

		return alpha

	def compute_generalization(self):
		if self.model_layers is None:
			layer = self.find_final_model_layer()
		else:
			layer = self.model_layers[-1]

		layerHandler = LayerHandler(self.model, self.loader, \
			layer, self.apply_dim_reduction)
		layer_features = layerHandler()

		score = self.run_smoothness_on_features(layer_features)
		print(f"generalization score for model is: {score}")
		






