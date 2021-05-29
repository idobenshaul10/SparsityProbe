from tree_models.random_forest import WaveletsForestRegressor

def train_model(x, y, mode='regression', trees=5, depth=9, features='auto',
				state=2000, nnormalization='volume'):
	
	model = WaveletsForestRegressor(mode=mode, trees=trees, depth=depth, features=features, \
		seed=state, norms_normalization=nnormalization)
	
	model.fit(x, y)
	return model


def run_alpha_smoothness(X, y, t_method='RF', n_trees=1, m_depth=9,
						 n_features='auto', n_state=2000, \
						 norm_normalization='volume', \
						 text='', output_folder='', epsilon_1=None, epsilon_2=None):	

	norm_m_term = 0    
	model = train_model(X, y, trees=n_trees, \
		depth=m_depth, features=n_features, state=n_state, \
		nnormalization=norm_normalization)	

	alpha = model.evaluate_angle_smoothness(text=text, \
		output_folder=output_folder, epsilon_1=epsilon_1, epsilon_2=epsilon_2)	

	return alpha