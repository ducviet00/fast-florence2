import os
from glob import glob

from safetensors import safe_open
from torch import nn


def load_model(model: nn.Module, path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param = model.get_parameter(weight_name)
                param.data.copy_(f.get_tensor(weight_name))
