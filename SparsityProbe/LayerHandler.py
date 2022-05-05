import torch
from tqdm import tqdm
from SparsityProbe.DimReducer import DimensionalityReducer
from dataclasses import dataclass
import numpy as np


@dataclass
class FeaturesBuffer:
    features: np.array = None
    shape: tuple = None

    def __call__(self, new_features: np.array, batch_size: int = None) -> None:
        # print(f"In feature buffer, new_features:{new_features.shape}")
        if self.features is not None:
            try:
                new_features = new_features.reshape(-1, self.shape[1])
            except:
                import pdb; pdb.set_trace()
            self.features = np.concatenate((self.features, new_features), axis=0)
        else:
            if batch_size is not None:
                new_features = new_features.reshape(batch_size, -1)
            self.features = new_features
        self.shape = self.features.shape
        # print(f"current buffer shape:{self.shape}")

    def clear_buffer(self):
        self.features = None


@dataclass
class LayerHandler:
    '''Allows to run model up to layer,
    and output different statistics
    '''
    model: torch.nn.Module
    loader: torch.utils.data.DataLoader
    layer: torch.nn.Module
    apply_dim_reduction: bool
    layer_features: FeaturesBuffer = None
    layer_features_buffer: FeaturesBuffer = None
    dim_reducer: DimensionalityReducer = None
    batch_size: int = None
    use_cuda: bool = None
    dim_reducer_section_count: int = -1
    handle: torch.utils.hooks.RemovableHandle = None

    def __post_init__(self):
        if self.model is not None:
            self.model.eval()
        self.layer_features = FeaturesBuffer()
        self.layer_features_buffer = FeaturesBuffer()
        self.batch_size = self.loader.batch_size
        self.use_cuda = torch.cuda.is_available()
        if self.apply_dim_reduction:
            self.dim_reducer = DimensionalityReducer()
            self.dim_reducer_section_count = self.dim_reducer.compute_section_number(len(self.loader.dataset))


    def get_activation(self):
        def hook(model, input, output):
            output = output[0] if (type(output) == tuple) else output
            new_outputs = output.detach().cpu().numpy()
            self.layer_features_buffer(new_outputs, batch_size=self.batch_size)
        return hook

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.get_activation())
        return self

    def __call__(self):
        data = None
        # for idx, loader_data in tqdm(enumerate(self.loader), total=len(self.loader)):
        for idx, loader_data in enumerate(self.loader):
            if type(loader_data) == dict:
                target = loader_data['labels']

                if 'pixel_values' in loader_data:
                    data = loader_data['pixel_values']
                    self.model(**{'pixel_values': data.cuda(), 'labels': target.cuda()})
                else:
                    data = loader_data['input_ids']
                    self.model(**{'input_ids': data.cuda(), 'labels': target.cuda()})
                # self.model(**{'pixel_values': data.cuda(), 'labels': target.cuda()})

            elif type(loader_data) == tuple or type(loader_data) == list:
                data, target = loader_data
                if self.use_cuda:
                    data = data.cuda()
                self.model(data)
            else:
                self.model(**loader_data)

            if self.apply_dim_reduction:
                if min(self.layer_features_buffer.shape) >= self.dim_reducer.threshold_dimension or (
                        (idx == len(self.loader) - 1) and self.dim_reducer.DR_fitted):

                    if not idx == len(self.loader) - 1:
                        if self.dim_reducer.counter == self.dim_reducer_section_count - 1:
                            continue

                    reduced_buffer_content = self.dim_reducer(self.layer_features_buffer.features)
                    # print(f"after dim reducer, reduced_buffer_content:{reduced_buffer_content.shape}")
                    self.layer_features(reduced_buffer_content)
                    self.layer_features_buffer.clear_buffer()
            if data is not None:
                del data

        if (not self.apply_dim_reduction) or (self.layer_features.features is None):
            layer_features = self.layer_features_buffer.features
        else:
            layer_features = self.layer_features.features

        return layer_features

    def __exit__(self, type, value, traceback):
        self.handle.remove()
        self.layer_features.clear_buffer()
        self.layer_features_buffer.clear_buffer()
        self.layer_features = None
        self.layer_features_buffer = None

