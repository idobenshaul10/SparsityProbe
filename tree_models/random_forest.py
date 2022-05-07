import os
import numpy as np
import logging
from sklearn import tree, linear_model, ensemble
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from tqdm import tqdm
from sklearn import preprocessing
from scipy.sparse import csr_matrix, lil_matrix


class WaveletsForestRegressor:
	def __init__(self, regressor='random_forest', mode='classification',
				 criterion='gini', depth=9, trees=5, seed=2000,
				 norms_normalization='volume', train_vi=True, vi_threshold=0.8):
		'''
        Construct a new 'WaveletsForestRegressor' object.
        :criterion: Splitting criterion. Same options as sklearn\'s DecisionTreeRegressor. Default is "mse".
        :depth: Maximum depth of each tree. Default is 9.
        :trees: Number of trees in the forest. Default is 5.
        :seed: Seed for random operations. Default is 2000.
        '''

		np.random.seed(seed)

		self.norms = None
		self.sorted_norms = None
		self.vals = None
		self.power = 2
		##
		self.volumes = None
		self.X = None
		self.y = None
		self.rf = None

		self.mode = mode
		self.regressor = regressor
		self.criterion = criterion
		self.verbose = False
		self.num_alpha_sample_points = 100
		self.depth = None if depth == -1 else depth

		self.trees = trees
		self.seed = seed
		self.norms_normalization = norms_normalization
		self.save_errors = False

		self.train_vi = train_vi
		self.vi_threshold = vi_threshold
		self.norms_normalization = norms_normalization

	def from_label_to_one_hot_label(self, y):
		if len(y.shape) > 1 and y.shape[1] != 1:
			return y
		num_samples = y.shape[0]
		self.num_classes = y.max() + 1
		# y_result = np.zeros((num_samples, self.num_classes))
		if self.verbose:
			print("creating one-hot encodings")
		y_result = np.eye(self.num_classes)[y]
		return y_result

	def fit(self, X_raw, y):
		'''
        :X_raw: Non-normalized features, given as a 2D array with each row representing a sample.
        :y: Labels, each row is given as a vertex on the simplex.
        '''
		logging.info('Fitting %s samples' % np.shape(X_raw)[0])
		if type(X_raw) == np.ndarray:
			try:
				min_max_scaler = preprocessing.MinMaxScaler()
				X = min_max_scaler.fit_transform(X_raw)
			except:
				X = (X_raw - np.min(X_raw, 0)) / ((np.max(X_raw, 0) - np.min(X_raw, 0)) + 1e-6)
			self.X = X
		else:
			X = (X_raw - X_raw.min()) / ((X_raw.max() - X_raw.min()) + 1e-6)
			self.X = X.cpu().numpy()

		self.num_classes = y.max() + 1
		if self.mode == 'classification':
			self.y = self.from_label_to_one_hot_label(y)
			del y
		else:
			self.y = y

		regressor = None

		# elif self.mode == 'regression':
		regressor = ensemble.RandomForestRegressor(
			# criterion="absolute_error",
			n_estimators=self.trees,
			max_depth=self.depth,
			max_features='auto',
			n_jobs=-1,
			random_state=self.seed,
			verbose=0
		)

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
		self.si_tree = np.zeros((np.shape(X)[1], len(rf.estimators_)))
		self.si = np.array([])
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
			num_samples = np.sum(paths_fullmat, 0) / paths_fullmat.shape[0]

			if self.norms_normalization == 'volume':
				norms = np.multiply(norms, np.power(volumes, 1 / self.power))
			else:
				norms = np.multiply(norms, np.power(num_samples, 1 / self.power))

			self.volumes = np.append(self.volumes, volumes)
			self.norms = np.append(self.norms, norms)
			if len(self.root_nodes) == 0:
				self.root_nodes.append(0)
			else:
				self.root_nodes.append(self.root_nodes[-1] + num_nodes)

			self.num_samples = np.append(self.num_samples, num_samples)
			self.vals = np.append(self.vals, vals, axis=1)
			##
			if self.train_vi:
				for k in range(0, num_features):
					vi_node_box = np.zeros((num_nodes, num_features, 2))
					vi_node_box[:, :, 1] = 1
					vi_norms = np.zeros(num_nodes)
					vi_vals = np.zeros((val_size, num_nodes))
					self.__variable_importance(estimator, 0, vi_node_box, vi_norms, vi_vals, k, self.vi_threshold)
					if self.norms_normalization == 'volume':
						vi_norms = np.multiply(vi_norms, np.power(volumes, 1 / self.power))
					else:
						vi_norms = np.multiply(vi_norms, np.power(num_samples, 1 / self.power))
					self.si_tree[k, i] = np.sum(vi_norms)

		import pdb; pdb.set_trace()
		self.si = np.append(self.si, np.sum(self.si_tree, 1) / len(rf.estimators_))
		self.feature_importances_ = self.si

		self.sorted_norms = np.argsort(-self.norms)
		return self

	def __compute_norm(self, avg, parent_avg, volume):
		norm = np.power(np.sum(np.power(np.abs(avg - parent_avg), self.power)) * volume, (1 / self.power))
		return norm

	def compute_average_score_from_tree(self, tree_value):
		return tree_value[:, 0]

	def __traverse_nodes(self, estimator, base_node_id, node_box, norms, vals, levels):
		if base_node_id == 0:
			vals[:, base_node_id] = self.compute_average_score_from_tree( \
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
				[estimator.tree_.threshold[base_node_id],
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

	def evaluate_angle_smoothness(self, text='', output_folder='', epsilon_1=None, epsilon_2=None,
								  compute_using_index=True):
		"""
        Evaluate smoothness using sparsity consideration
        """
		mask = np.ones(len(self.norms), dtype=bool)
		mask[self.root_nodes] = False
		norms = self.norms[mask]
		h = 0.01
		J = len(self.rf.estimators_)

		if not compute_using_index:
			alpha = 1.
			tau = 1 / (alpha + 1 / self.power)
			tau_sparsity = (1 / J) * np.power(np.power(norms, tau).sum(), (1 / tau))
			return np.array([tau_sparsity])
		else:
			diffs = []
			taus = np.arange(1e-3, self.power, h)
			for tau in tqdm(taus):
				tau_sparsity = (1 / J) * np.power(np.power(norms, tau).sum(), ((1 / tau) - 1))
				tau_sparsity *= np.power(norms, (tau - 1)).sum()
				diffs.append(tau_sparsity)
			diffs = -np.array(diffs)
			angles = np.rad2deg(np.arctan(diffs))

			try:
				assert (epsilon_2 > epsilon_1)
				step = (epsilon_2 - epsilon_1) / self.num_alpha_sample_points
				sampling_indices = np.arange(epsilon_1, epsilon_2, step)
				alphas = []
				for idx, sample_epsilon in enumerate(sampling_indices):
					cur_epsilon_indices = np.where(abs(angles + 90.) <= sample_epsilon)[0]
					if idx == 0:
						epsilon_1_indices = cur_epsilon_indices
					elif idx == len(sampling_indices) - 1:
						epsilon_2_indices = cur_epsilon_indices
					cur_angle_index = cur_epsilon_indices[-1]
					cur_critical_tau = taus[cur_angle_index]
					cur_critical_alpha = ((1 / cur_critical_tau) - 1 / self.power)
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
				plt.show(block=True)
				print(f"abs(angles+90.).min():{abs(angles + 90.).min()}")

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
				plt.show(block=True)
				plt.clf()
			return alphas

	def predict(self, X, m=1000, start_m=0, paths=None):
		'''
		Predict using a maximum of M-terms

		:X: Data samples.
		:m: Maximum of M-terms.
		:start_m: The index of the starting term. Can be used to evaluate predictions incrementally over terms.
		:paths: Instead of computing decision paths for each sample, the method can receive the indicator matrix. Can be used to evaluate predictions incrementally over terms.
		:return: Predictions.
		'''

		cur_sorted_norms = self.sorted_norms[start_m: m]
		if paths == None:
			paths, _ = self.rf.decision_path(X)
		pruned = lil_matrix(paths.shape, dtype=np.float32)
		pruned[:, cur_sorted_norms] = paths[:, cur_sorted_norms]
		predictions = pruned * self.vals.T / len(self.rf.estimators_)
		return predictions, paths

	def __variable_importance(self, estimator, base_node_id, vi_node_box, vi_norms, vi_vals, feature, threshod):
		if base_node_id == 0:
			vi_vals[:, base_node_id] = estimator.tree_.value[base_node_id][:, 0]
			vi_norms[base_node_id] = self.__compute_norm(vi_vals[:, base_node_id], 0, 1)

		left_id = estimator.tree_.children_left[base_node_id]
		right_id = estimator.tree_.children_right[base_node_id]
		if left_id >= 0:
			vi_node_box[left_id, :, :] = vi_node_box[base_node_id, :, :]
			vi_node_box[left_id, estimator.tree_.feature[base_node_id], 1] = np.min(
				[estimator.tree_.threshold[base_node_id],
				 vi_node_box[left_id, estimator.tree_.feature[base_node_id], 1]])
			self.__variable_importance(estimator, left_id, vi_node_box, vi_norms, vi_vals, feature, self.vi_threshold)
			vi_vals[:, left_id] = estimator.tree_.value[left_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
			tnorm = self.__compute_norm(vi_vals[:, left_id], vi_vals[:, base_node_id], 1)
			if estimator.tree_.feature[estimator.tree_.children_left[base_node_id]] == feature and tnorm > threshod:
				vi_norms[left_id] = tnorm
		if right_id >= 0:
			vi_node_box[right_id, :, :] = vi_node_box[base_node_id, :, :]
			vi_node_box[right_id, estimator.tree_.feature[base_node_id], 0] = np.max(
				[estimator.tree_.threshold[base_node_id],
				 vi_node_box[right_id, estimator.tree_.feature[base_node_id], 0]])
			self.__variable_importance(estimator, right_id, vi_node_box, vi_norms, vi_vals, feature, self.vi_threshold)
			vi_vals[:, right_id] = estimator.tree_.value[right_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
			tnorm = self.__compute_norm(vi_vals[:, right_id], vi_vals[:, base_node_id], 1)
			if estimator.tree_.feature[estimator.tree_.children_right[base_node_id]] == feature and tnorm > threshod:
				vi_norms[right_id] = tnorm
