from tqdm import tqdm
from utils import *
import time
from environments.base_environment import *
from torchvision import datasets, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import wandb


class cifar10_vit(BaseEnviorment):
    def __init__(self):
        super().__init__()
        self.model_name = 'nateraw/vit-base-patch16-224-cifar10'
        # self.model_name = 'facebook/deit-tiny-patch16-224'
        # self.model_name = 'facebook/dino-vits8'


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
        feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        transform = transforms.Compose([
            transforms.Resize((feature_extractor.size, feature_extractor.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=feature_extractor.image_mean,
                                 std=feature_extractor.image_std)
        ])
        return transform

    def get_layers(self, model):
        return []

    def get_model(self, **kwargs):
        model = ViTForImageClassification.from_pretrained(self.model_name)
        wandb.config.update({"model": self.model_name})
        return model
