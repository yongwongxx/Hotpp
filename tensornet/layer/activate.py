import torch
import torch.nn.functional as F
from .base import TensorActivateLayer
from ..utils import expand_to


class TensorTanh(TensorActivateLayer):

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.tanh(input_tensor)

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        norm = self.weights * torch.sum(input_tensor ** 2, dim=tuple(range(2, 2 + way))) + self.bias
        factor = F.tanh(norm) / torch.where(norm == 0, torch.ones_like(norm), norm)
        return expand_to(factor, 2 + way) * input_tensor


class TensorRelu(TensorActivateLayer):

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.relu(input_tensor)
    
    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        norm = self.weights * torch.sum(input_tensor ** 2, dim=tuple(range(2, 2 + way))) + self.bias
        factor = F.relu(norm) / torch.where(norm == 0, torch.ones_like(norm), norm)
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
        norm = self.weights * torch.sum(input_tensor ** 2, dim=tuple(range(2, 2 + way))) + self.bias
        factor = F.sigmoid(norm)
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
        norm = self.weights * torch.sum(input_tensor ** 2, dim=tuple(range(2, 2 + way))) + self.bias
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
