from tqdm import tqdm
from utils import *
import time
from environments.base_environment import *
from torchvision import datasets, transforms
from models.resnet import resnet18


class cifar10_env(BaseEnviorment):
    def __init__(self):
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def get_dataset(self):
        dataset = datasets.CIFAR10(root=r'cifar10',
                                   train=True,
                                   transform=self.get_eval_transform(),
                                   download=True)
        return dataset

    def get_test_dataset(self):
        dataset = datasets.CIFAR10(root=r'cifar10',
                                   train=False,
                                   transform=self.get_eval_transform(),
                                   download=True)
        return dataset

    def get_eval_transform(self):
        transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize(self.mean, self.std)])
        return transform

    def get_layers(self, model):
        layers = [model.layer1, model.layer2, model.layer3]
        final_layers = []
        for layer in layers:
            final_layers.append(layer[0]._modules['conv1'])
            final_layers.append(layer[0]._modules['conv2'])
        final_layers.append(model.avgpool)

        return final_layers

    def get_model(self, **kwargs):
        model = resnet18(pretrained=True)
        if self.use_cuda:
            model = model.cuda()

        return model
