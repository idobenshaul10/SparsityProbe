

# Sparsity Probe: Analysis tool for Deep Learning Models

[![GitHub license](https://img.shields.io/github/license/idobenshaul10/SparsityProbe)](https://github.com/idobenshaul10/SparsityProbe/blob/main/LICENSE)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-Pytorch-1f425f.svg)](https://pytorch.org/)

This repository is a limited implementation of  [Sparsity Probe: Analysis tool for Deep Learning Models](https://arxiv.org/abs/2105.06849) by I. Ben-Shaul and S. Dekel (2021).

![Folded Ball Example](/Images/folding_ball.png)

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
The first step of using this Repo should be to look at this example: [CIFAR10 Example](https://github.com/idobenshaul10/SparsityProbe/blob/main/examples/Example%20CIFAR10.ipynb).
In this example, we demonstrate running the Sparsity-Probe on a trained Resnet18 on the CIFAR10 dataset, at selected layers. 

### Creating a new enviorment:
Create a new environment in the `environments` directory, inheriting from `BaseEnviorment`. This enviorment should include the train and test datasets(including the matching transforms), the model layers we want to test the alpha-scores on(see `cifar10_env` example), and the trained model.

### Training a model:
It is possible to train a basic model with the [train.py](https://github.com/idobenshaul10/SparsityProbe/blob/main/train/train.py) script, which uses an environment to load the model and the datasets. 
*Example Usage:*
`python train/train_mnist.py --output_path "results" --batch_size 32 --epochs 100`

### Running the Sparsity Probe
Done using the [DL_smoothness.py](https://github.com/idobenshaul10/SparsityProbe/blob/main/DL_Layer_Analysis/DL_smoothness.py) script. 
**Arguments:**<br />
`trees` - Number of trees in the forest.<br />
`depth` - Maximum depth of each tree.<br />
`batch_size` - batch used in the forward pass(when computing the layer outputs)<br />
`env_name` - enviorment which is loaded to measure alpha-scores on <br />
`epsilon_1` - the epsilon_low used for the numerical approximation. By default, epsilon_high is 
inited as 4*epsilon_low<br />
`only_umap` - only create umaps of the intermediate layers(without computing alpha-scores)<br />
`use_clustering` - run KMeans on intermediate layers<br />
`calc_test` - calculate test accuracy(More metrics coming soon)	<br />
`output_folder` - location where all outputs are saved<br />
`feature_dimension` - to reduce computation costs, we compute the alpha-scores on the features after a dimensionality reduction technique has been applied. As of now, if the dim(layer_outputs)>feature_dimension, the TruncatedSVD is used to 
reduce dim(layer_outputs) to feature_dimension. Default feature_dimension is 2500.<br />
### Plotting Results
Result plots can be created using [this](https://github.com/idobenshaul10/SparsityProbe/blob/main/DL_Layer_Analysis/plot_DL_json_results.py) script. 

![UMAP example](/Images/umap_cifar10.jpg)


## Acknowledgements
Our pretrained CIFAR10 Resnet18 network used in the example is taken from [This Repo](https://github.com/huyvnphan/PyTorch_CIFAR10). 

## License

This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
