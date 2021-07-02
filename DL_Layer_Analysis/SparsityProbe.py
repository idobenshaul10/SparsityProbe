from tqdm import tqdm
import torch
from torch import nn
from DL_Layer_Analysis.LayerHandler import LayerHandler
from DL_Layer_Analysis.ModelHandler import ModelHandler
from tree_models.random_forest import WaveletsForestRegressor

class SparsityProbe():
	def __init__(self, loader, model, model_layers=None, \
		apply_dim_reduction=False, epsilon_1=0.1, epsilon_2=0.4):
		self.loader = loader
		self.model = model
		self.model_handler = ModelHandler(model)
		self.model_layers = model_layers
		self.apply_dim_reduction = apply_dim_reduction
		self.labels = self.get_labels()		
		
		# tree parameters
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

	def aggregate_scores(self, scores):
		'''at the moment we use mean aggregation for alpha scores'''
		return scores.mean()

	def train_tree_model(self, x, y, mode='regression', \
		trees=5, depth=9, features='auto',
		state=2000, nnormalization='volume'):

		model = WaveletsForestRegressor(mode=mode, trees=trees, depth=depth, features=features, \
			seed=state, norms_normalization=nnormalization)

		model.fit(x, y)
		return model
	
	def run_smoothness_on_features(self, features):		
		tree_model = self.train_tree_model(features, self.labels, \
			trees=self.n_trees, depth=self.depth, features=self.n_features, \
			state=self.n_state, nnormalization=self.norm_normalization, \
			mode='classification')

		alpha = tree_model.evaluate_angle_smoothness(text='try', \
			output_folder=self.output_folder, epsilon_1=self.epsilon_1, \
			epsilon_2=self.epsilon_2)

		alpha = self.aggregate_scores(alpha)
		return alpha

	def compute_generalization(self):
		if self.model_layers is None:
			layer = self.model_handler.get_final_layer()
		else:
			layer = self.model_layers[-1]

		layerHandler = LayerHandler(self.model, self.loader, \
			layer, self.apply_dim_reduction)
		layer_features = layerHandler()
		score = self.run_smoothness_on_features(layer_features)		
		print(f"generalization score for model is: {score}")
		






