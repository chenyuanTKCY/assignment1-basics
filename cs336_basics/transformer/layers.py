import torch
import torch.nn as nn
import einops
import math
from jaxtyping import Float

class Linear(nn.Module):
    """ Linear Layer"""

    def __init__(self, in_features: int , out_features: int, device :torch.device | None = None, dtype : torch.dtype|None = None) -> None:
        # super().__init__(*args, **kwargs)
        """
        Initialize the linear layer.
        in_features: int stands for the input features.
        out_features: int stands for the output features.
        dtype: data type of the weight matirx
        """
        super().__init__()
        self.weight =  nn.Parameter(torch.zeros(out_features,in_features, device = device, dtype = dtype))
        stdev = (2/(in_features+out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, 0, stdev, -3 * stdev, 3 * stdev )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T