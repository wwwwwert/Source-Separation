import torch
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.main_body = nn.Sequential(
            nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dims),
            nn.PReLU(),
            nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dims),
            nn.PReLU()
        )
        
        self.use_residual_conv = (in_dims != out_dims)
        if self.use_residual_conv:
            self.residual_conv = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.prelu = nn.PReLU()
        self.pooling = nn.MaxPool1d(3)

    def forward(self, x: torch.Tensor):
        residual = x
        if self.use_residual_conv: 
            residual = self.residual_conv(x)
        x = self.main_body(x)
        x = x + residual
        x = self.prelu(x)
        x = self.pooling(x)
        return x