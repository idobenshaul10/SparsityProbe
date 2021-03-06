import os
import numpy as np
import logging
from sklearn import tree, linear_model, ensemble
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from tqdm import tqdm

class WaveletsForestRegressor:
	def __init__(self, regressor='random_forest', mode='classification', \
				criterion='gini', train_vi=False,
				depth=9, trees=5, features='auto', seed=2000, \
				norms_normalization='volume'):
		'''
		Construct a new 'WaveletsForestRegressor' object.		
		:criterion: Splitting criterion. Same options as sklearn\'s DecisionTreeRegressor. Default is "mse".		
		:depth: Maximum depth of each tree. Default is 9.
		:trees: Number of trees in the forest. Default is 5.
		:features: Features to consider in each split. Same options as sklearn\'s DecisionTreeRegressor.
		:seed: Seed for random operations. Default is 2000.
		'''

		np.random.seed(2000)

		self.norms = None
		self.vals = None
		self.power = 2		
		self.feature_importances_ = None
		##
		self.volumes = None
		self.X = None
		self.y = None
		self.rf = None

		self.mode = mode
		self.regressor = regressor
		self.criterion = criterion		
		self.verbose = False

		self.num_alpha_sample_points = 10
		self.depth = None if depth == -1 else depth

		self.trees = trees
		self.seed = seed		
		self.norms_normalization = norms_normalization
		self.save_errors = False
	
	def from_label_to_one_hot_label(self, y):
		if y.shape[1] != 1:
			return y
		num_samples = y.shape[0]
		self.num_classes = y.max()+1
		y_result = np.zeros((num_samples, self.num_classes))
		for i in range(num_samples):			
			y_result[i][y[i][0]] = 1.		
		return y_result

	def fit(self, X_raw, y):
		'''
		:X_raw: Non-normalized features, given as a 2D array with each row representing a sample.
		:y: Labels, each row is given as a vertex on the simplex.
		'''		
		logging.info('Fitting %s samples' % np.shape(X_raw)[0])				
		if type(X_raw) == np.ndarray:			
			X = (X_raw - np.min(X_raw, 0))/(np.max(X_raw, 0) - np.min(X_raw, 0))
			self.X = X
		else:
			X = (X_raw - X_raw.min())/(X_raw.max() - X_raw.min())
			self.X = X.cpu().numpy()

		self.num_classes = y.max()+1
		self.y = self.from_label_to_one_hot_label(y)

		regressor = None		
		if self.mode == 'classification':
			regressor = ensemble.RandomForestClassifier(
				criterion='gini',
				n_estimators=self.trees, 
				max_depth=self.depth,
				max_features='auto',
				n_jobs=-1,
				random_state=self.seed,
			)				

		elif self.mode == 'regression':
			regressor = ensemble.RandomForestRegressor(
				n_estimators=self.trees, 
				max_depth=self.depth,
				max_features='auto',
				n_jobs=-1,
				random_state=self.seed,
				verbose=2
			)				
		else:
			print("ERROR, WRONG MODE")
			exit()
		
		try:			
			rf = regressor.fit(self.X, self.y.ravel())
		except Exception as e:
			rf = regressor.fit(self.X, self.y)
			
		self.rf = rf

		try:
			val_size = np.shape(self.y)[1]
		except:
			val_size = 1

		self.norms = np.array([])
		self.vals = np.zeros((val_size, 0))
		self.volumes = np.array([])		
		self.num_samples = np.array([])		
		self.root_nodes = []		

		for i in range(len(rf.estimators_)):
			estimator = rf.estimators_[i]
			num_nodes = len(estimator.tree_.value)
			num_features = np.shape(X)[1]
			node_box = np.zeros((num_nodes, num_features, 2))
			node_box[:, :, 1] = 1

			norms = np.zeros(num_nodes)
			vals = np.zeros((val_size, num_nodes))			
			levels = np.zeros((norms.shape[0]))

			self.__traverse_nodes(estimator, 0, node_box, norms, vals, levels)			

			volumes = np.product(node_box[:, :, 1] - node_box[:, :, 0], 1)

			paths = estimator.decision_path(X)
			paths_fullmat = paths.todense()
			num_samples = np.sum(paths_fullmat, 0)/paths_fullmat.shape[0]

			if self.norms_normalization == 'volume':
				norms = np.multiply(norms, np.power(volumes, 1/self.power))
			else:
				norms = np.multiply(norms, np.power(num_samples, 1/self.power))


			self.volumes = np.append(self.volumes, volumes)
			self.norms = np.append(self.norms, norms)
			if len(self.root_nodes) == 0:
				self.root_nodes.append(0)
			else:
				self.root_nodes.append(self.root_nodes[-1] + num_nodes)

			self.num_samples = np.append(self.num_samples, num_samples)
			self.vals = np.append(self.vals, vals, axis=1)			
		
		return self

	def __compute_norm(self, avg, parent_avg, volume):		
		norm = np.power(np.sum(np.power(np.abs(avg - parent_avg), self.power)) * volume, (1/self.power))
		return norm

	def compute_average_score_from_tree(self, tree_value):		
		if self.mode == 'classification':			
			y_vec = [-1. , 1.]
			result = tree_value.dot(y_vec)/tree_value.sum()         
			return result
		else:
			return tree_value[:, 0]

	def __traverse_nodes(self, estimator, base_node_id, node_box, norms, vals, levels):

		if base_node_id == 0:
			vals[:, base_node_id] = self.compute_average_score_from_tree(\
				estimator.tree_.value[base_node_id])
			norms[base_node_id] = self.__compute_norm(vals[:, base_node_id], 0, 1)			

		left_id = estimator.tree_.children_left[base_node_id]
		right_id = estimator.tree_.children_right[base_node_id]


		if left_id >= 0:			
			
			levels[left_id] = levels[base_node_id] + 1
			tree = estimator.tree_			
			left_feature = tree.feature[base_node_id]
			left_threshold = tree.threshold[base_node_id]

			node_box[left_id, :, :] = node_box[base_node_id, :, :]			
			node_box[left_id, estimator.tree_.feature[base_node_id], 1] = np.min(
				[estimator.tree_.threshold[base_node_id], \
				node_box[left_id, estimator.tree_.feature[base_node_id], 1]])
			self.__traverse_nodes(estimator, left_id, node_box, norms, vals, levels)
			vals[:, left_id] = self.compute_average_score_from_tree(estimator.tree_.value[left_id]) - \
				self.compute_average_score_from_tree(estimator.tree_.value[base_node_id])
			norms[left_id] = self.__compute_norm(vals[:, left_id], vals[:, base_node_id], 1)

		if right_id >= 0:

			levels[right_id] = levels[base_node_id] + 1
			tree = estimator.tree_
			right_feature = tree.feature[base_node_id]
			right_threshold = tree.threshold[base_node_id]			

			node_box[right_id, :, :] = node_box[base_node_id, :, :]
			node_box[right_id, estimator.tree_.feature[base_node_id], 0] = np.max(
				[estimator.tree_.threshold[base_node_id], node_box[right_id, estimator.tree_.feature[base_node_id], 0]])
			self.__traverse_nodes(estimator, right_id, node_box, norms, vals, levels)
			vals[:, right_id] = self.compute_average_score_from_tree(estimator.tree_.value[right_id]) - \
				self.compute_average_score_from_tree(estimator.tree_.value[base_node_id])
			norms[right_id] = self.__compute_norm(vals[:, right_id], vals[:, base_node_id], 1)
	
	def evaluate_angle_smoothness(self, m=1000, error_TH=0, text='', \
			output_folder='', epsilon_1=None, epsilon_2=None):
		'''
		Evaluate smoothness using sparsity consideration
		'''
		approx_diff = False		
		mask = np.ones(len(self.norms), dtype=bool)
		mask[self.root_nodes] = False
		norms = self.norms[mask]
		h = 0.01
		
		diffs = []		
		taus = np.arange(0.3, 2., h)
		total_sparsities, total_alphas = [], []
		J = len(self.rf.estimators_)

		use_derivatives = True
		for tau in tqdm(taus):
			if use_derivatives:
				tau_sparsity = (1/J)*np.power(np.power(norms, tau).sum(), ((1/tau)-1))
				tau_sparsity *= np.power(norms, (tau-1)).sum()
			else:
				tau_sparsity = (1/J)*np.power(np.power(norms, tau).sum(), (1/tau))
			diffs.append(tau_sparsity)
		diffs = -np.array(diffs)


		angles = np.rad2deg(np.arctan(diffs))
		try:
			step = (epsilon_2 - epsilon_1)/self.num_alpha_sample_points
			sampling_indices = np.arange(epsilon_1, epsilon_2, step)
			alphas = []
			for idx, sample_epsilon in enumerate(sampling_indices):				
				cur_epsilon_indices = np.where(abs(angles+90.)<=sample_epsilon)[0]
				if idx == 0:
					epsilon_1_indices = cur_epsilon_indices
				elif idx == len(sampling_indices)-1:
					epsilon_2_indices = cur_epsilon_indices
				cur_angle_index = cur_epsilon_indices[-1]				
				cur_critical_tau = taus[cur_angle_index]
				cur_critical_alpha = ((1/cur_critical_tau) - 1/self.power)
				alphas.append(cur_critical_alpha)
			alphas = np.array(alphas)				

		except Exception as e:
			print(f"\nHIGH RANGE EPSILON:{epsilon_1} indices are empty, try considering a bigger EPSILON")
			exit()
		
		if self.verbose:
			colors = ["#2bc2cb", "#1b374d", "#ee4f2f", "#fba720"]
			plt.figure(1)		
			plt.title(f"tau vs. angle")
			plt.xlabel(f'tau')
			plt.ylabel(f'sparsity angle')
			plt.plot(taus, angles, zorder=1, color=colors[0], label='sparsity derivative angle')
			plt.scatter(taus[epsilon_2_indices], angles[epsilon_2_indices], \
				color=colors[1], zorder=2, s=0.5, label='epsilon low')
			plt.scatter(taus[epsilon_1_indices], angles[epsilon_1_indices], \
				color=colors[2], zorder=2, s=0.5, label='epsilon high')			

			plt.legend()

			print(f"abs(angles+90.).min():{abs(angles+90.).min()}")

			save_path = os.path.join(output_folder, f"{text}_{epsilon_1}_{epsilon_2}_derivates.png")
			print(f"save_path:{save_path}")
			plt.savefig(save_path, \
				dpi=300, bbox_inches='tight')
			plt.clf()

			plt.figure(2)
			plt.title(f"tau vs. derivative")
			plt.xlabel(f'tau')
			plt.ylabel(f'sparsity derivative')			
			plt.plot(taus, diffs, zorder=1, color=colors[0], label='sparsity derivative')
			plt.scatter(taus[epsilon_2_indices], diffs[epsilon_2_indices], \
				color=colors[1], zorder=2, s=0.5, label='epsilon low')
			plt.scatter(taus[epsilon_1_indices], diffs[epsilon_1_indices], \
				color=colors[2], zorder=2, s=0.5, label='epsilon high')

			plt.legend()

			save_path = os.path.join(output_folder, f"{text}_{epsilon_1}_{epsilon_2}_angles.png")
			print(f"save_path:{save_path}")
			plt.savefig(save_path, \
				dpi=300, bbox_inches='tight')
			plt.clf()
		
		return alphas
	
	