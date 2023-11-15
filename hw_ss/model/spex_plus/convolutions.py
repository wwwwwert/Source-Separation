import torch
from torch import nn 

from hw_ss.model.spex_plus.layers import GlobalLayerNorm
    

class TCNBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_dim: int=128,
            kernel_size: int=3,
            dilation: int=1
        ):
        super().__init__()
        self.conv_1d_1 = nn.Conv1d(in_channels, hidden_dim, 1)
        self.prelu1 = nn.PReLU()
        self.global_norm_1 = GlobalLayerNorm(hidden_dim)
        pad_size = (dilation * (kernel_size - 1)) // 2
        self.de_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            groups=hidden_dim,
            padding=pad_size,
            dilation=dilation,
            bias=True
        )
        self.prelu2 = nn.PReLU()
        self.global_norm_2 = GlobalLayerNorm(hidden_dim)
        self.conv_1d_2 = nn.Conv1d(hidden_dim, in_channels, 1, bias=True)


    def forward(self, x: torch.Tensor):
        y = self.conv_1d_1(x)
        y = self.global_norm_1(self.prelu1(y))
        y = self.de_conv(y)
        y = self.global_norm_2(self.prelu2(y))
        y = self.conv_1d_2(y)
        x = x + y
        return x
    

class FirstTCNBlock(nn.Module):
    def __init__(
            self,
            speaker_embedding_dim: int,
            in_channels: int,
            hidden_dim: int=128,
            kernel_size: int=3,
            dilation: int=1
        ):
        super().__init__()
        self.conv_1d_1 = nn.Conv1d(in_channels + speaker_embedding_dim, hidden_dim, 1)
        self.prelu1 = nn.PReLU()
        self.global_norm_1 = GlobalLayerNorm(hidden_dim)
        pad_size = (dilation * (kernel_size - 1)) // 2
        self.de_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            groups=hidden_dim,
            padding=pad_size,
            dilation=dilation,
            bias=True
        )
        self.prelu2 = nn.PReLU()
        self.global_norm_2 = GlobalLayerNorm(hidden_dim)
        self.conv_1d_2 = nn.Conv1d(hidden_dim, in_channels, 1, bias=True)

    def forward(self, x: torch.Tensor, ref_embeddings: torch.Tensor):
        T = x.shape[-1]
        ref_embeddings = ref_embeddings.unsqueeze(-1)
        ref_embeddings = ref_embeddings.repeat(1, 1, T)
        y = torch.cat([x, ref_embeddings], 1)
        y = self.conv_1d_1(y)
        y = self.global_norm_1(self.prelu1(y))
        y = self.de_conv(y)
        y = self.global_norm_2(self.prelu2(y))
        y = self.conv_1d_2(y)
        x = x + y
        return x