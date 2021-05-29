import torch

class BaseEnviorment():
	def __init__(self):		
		self.use_cuda = torch.cuda.is_available()

	def load_enviorment(self, **kwargs):
		train_dataset = self.get_dataset()
		test_dataset = None
		try:
			test_dataset = self.get_test_dataset()
		except:
			print("test_dataset not availbale!")
		model = self.get_model(**kwargs)
		if torch.cuda.is_available():
			model = model.cuda()
		layers = self.get_layers(model)

		return model, train_dataset, test_dataset, layers

	def get_layers(self):
		pass
	
	def get_dataset(self):
		pass
	
	def get_eval_transform(self):
		pass
	
	def get_model(self):
		pass
