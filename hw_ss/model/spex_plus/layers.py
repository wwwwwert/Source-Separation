import torch
from torch import nn


class ChannelLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        # N x C x T => N x T x C
        x = x.transpose(1, 2)

        x = super().forward(x)
        x = x.transpose(1, 2)
        return x
    

class GlobalLayerNorm(nn.Module):
    """Global normalization layer from paper. 
    Seems to be like regular layer norm, but elements are normalized by two dims.
    """
    def __init__(
            self, 
            dim: int 
        ):
        super().__init__()
        self.eps = 1e-05
        self.beta = nn.Parameter(torch.zeros(dim, 1))
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x