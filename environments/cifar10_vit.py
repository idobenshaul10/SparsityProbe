from tqdm import tqdm
from utils import *
import time
from environments.base_environment import *
from torchvision import datasets, transforms
from transformers import AutoFeatureExtractor, ViTForImageClassification
import wandb


class cifar10_vit(BaseEnviorment):
    def __init__(self):
        super().__init__()

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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

    def get_layers(self, model):
        return []

    def get_model(self, **kwargs):
        # model_name = 'nateraw/vit-base-patch16-224-cifar10'
        model_name = 'facebook/deit-tiny-patch16-224'
        model = ViTForImageClassification.from_pretrained(model_name)
        wandb.config.update({"model": model_name})
        return model
