import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import weight_norm

class LogitLaplaceLoss(nn.Module):
    """Implements the Logit-Laplace Loss.

    The Logit-Laplace Loss is used for training variational autoencoders
    with pixel values that lie within a bounded interval.

    Attributes:
        epsilon (float): Small constant used to avoid numerical problems.
    """

    def __init__(self, epsilon: float = 0.1):
        super(LogitLaplaceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, mu: torch.Tensor, ln_b: torch.Tensor) -> torch.Tensor:
        """Computes the Logit-Laplace Loss.

        Args:
            x (torch.Tensor): Input tensor with shape: 
                (batch_size, num_channels, height, width).
            mu (torch.Tensor): Mean of the Logit-Laplace distribution with shape: 
                (batch_size, num_channels, height, width).
            ln_b (torch.Tensor): Natural logarithm of the scale parameter 
                of the Logit-Laplace distribution with shape: 
                (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: The computed Logit-Laplace Loss with shape: ().
        """
        x_mapped = self.map_pixels(x)
        b = torch.exp(ln_b)
        return -torch.mean(1/(2*b*x_mapped*(1-x_mapped)) * torch.exp(-torch.abs(torch.logit(x_mapped) - mu)/b))

    def map_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms the pixel values of the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape: 
                (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Transformed input tensor with shape: 
                (batch_size, num_channels, height, width).

        Raises:
            ValueError: If input tensor is not 4D or is not of type float.
        """
        if len(x.shape) != 4:
            raise ValueError('expected input to be 4d')
        if x.dtype != torch.float:
            raise ValueError('expected input to have type float')

        return (1 - 2 * self.epsilon) * x + self.epsilon

    def unmap_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transforms the pixel values of the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape: 
                (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Inverse transformed input tensor with shape: 
                (batch_size, num_channels, height, width).

        Raises:
            ValueError: If input tensor is not 4D or is not of type float.
        """
        if len(x.shape) != 4:
            raise ValueError('expected input to be 4d')
        if x.dtype != torch.float:
            raise ValueError('expected input to have type float')

        return torch.clamp((x - self.epsilon) / (1 - 2 * self.epsilon), 0, 1)
    
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

# Adapted from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
class VICReg(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_features: int,
                 sim_coeff: float = 1.0,
                 std_coeff: float = 1.0,
                 cov_coeff: float = 1.0):
        super().__init__()
        sim_coeff = sim_coeff
        std_coeff = std_coeff
        cov_coeff = cov_coeff
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.projector = Projector(self.embed_dim, num_features)

    def forward(self, student_features, teacher_features):
        x = self.projector(student_features)
        y = self.projector(teacher_features)

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss
    
class Projector(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=2,
        hidden_dim=None,
        hidden_mult=4,
        mlp_bias=True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim * hidden_mult
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, 
                              in_dim, 
                              out_dim, 
                              hidden_dim=hidden_dim, 
                              use_bn=use_bn, 
                              bias=mlp_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
