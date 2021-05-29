
# Sparsity Probe: Analysis tool for Deep Learning Models

[![GitHub license](https://img.shields.io/github/license/idobenshaul10/SparsityProbe)](https://github.com/idobenshaul10/SparsityProbe/blob/main/LICENSE)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-Pytorch-1f425f.svg)](https://pytorch.org/)

This repository is a limited implementation of  [Sparsity Probe: Analysis tool for Deep Learning Models](https://arxiv.org/abs/2105.06849) by I. Ben-Shaul and S. Dekel (2021).


## Downloading the Repo
```
git clone https://github.com/idobenshaul10/SparsityProbe.git
pip install -r requirements.txt
```

## Requirements
```
torch==1.7.0
umap_learn==0.4.6
matplotlib==3.3.2
tqdm==4.49.0
seaborn==0.11.0
torchvision==0.8.1
numpy==1.19.2
scikit_learn==0.24.2
umap==0.1.1
```

## Usage
An example usage of running the Sparsity-Probe on a trained Neural Network is shown in [CIFAR10 Example](https://github.com/idobenshaul10/SparsityProbe/blob/main/examples/Example%20CIFAR10.ipynb).


## Acknowledgements
Our pretrained CIFAR10 Resnet18 network used in the example is taken from [This Repo](https://github.com/huyvnphan/PyTorch_CIFAR10). 

## License

This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
