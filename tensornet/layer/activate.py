# TODO: There are now in-place operations in TensorTanh and TensorRelu, why?
import torch
import torch.nn.functional as F
from .base import TensorActivateLayer
from ..utils import expand_to


class TensorTanh(TensorActivateLayer):

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return torch.tanh(input_tensor)

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        #norm = self.weights * torch.sum(input_tensor ** 2, dim=tuple(range(2, 2 + way))) + self.bias
        #nonzero_norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        #factor = torch.tanh(norm) / nonzero_norm
        #return expand_to(factor, 2 + way) * input_tensor
        norm = torch.sum(input_tensor ** 2, dim=tuple(way for way in range(2, 2 + way)))
        norm = self.weights * norm + self.bias
        factor = torch.tanh(norm) / torch.where(norm == 0, torch.ones_like(norm), norm)
        output_tensor = input_tensor * expand_to(factor, 2 + way)
        return output_tensor


class TensorRelu(TensorActivateLayer):

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.relu(input_tensor)
    
    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        input_tensor_ = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        norm = self.weights * torch.sum(input_tensor_ ** 2, dim=2) + self.bias
        factor = torch.where(norm == 0, torch.ones_like(norm), norm)
        return expand_to(factor, 2 + way) * input_tensor


class TensorSilu(TensorActivateLayer):
    """TensorSilu
    silu(x) = x * sigmoid(x), so we the factor should be F.sigmoid(norm)
    """

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.silu(input_tensor)

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        input_tensor_ = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        norm = self.weights * torch.sum(input_tensor_ ** 2, dim=2) + self.bias
        factor = torch.sigmoid(norm)
        return expand_to(factor, 2 + way) * input_tensor


class TensorJilu(TensorActivateLayer):
    """TensorJilu
    Similar to TensorSilu, but use use tanh(x) as factor so the factor could be negative
    """
    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.silu(input_tensor)

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        input_tensor_ = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        norm = self.weights * torch.sum(input_tensor_ ** 2, dim=2) + self.bias
        factor = F.tanh(norm)
        return expand_to(factor, 2 + way) * input_tensor
    

class TensorIdentity(TensorActivateLayer):
    """TensorIdentity
    """
    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return input_tensor

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        return input_tensor


TensorActivateDict = {
    "tanh": TensorTanh,
    "relu": TensorRelu,
    "silu": TensorSilu,
    "jilu": TensorJilu,
    "none": TensorIdentity,
}
