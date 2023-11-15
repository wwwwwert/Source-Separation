import torch
from torch import nn

from .convolutions import FirstTCNBlock, TCNBlock
from .layers import ChannelLayerNorm
from .resnet import ResNetBlock


class EncoderBlock(nn.Module):
    def __init__(
            self,
            out_channels: int=256,
            kernel_short_length: int=20,
            kernel_middle_length: int=80,
            kernel_long_length: int=160
        ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_short_length = kernel_short_length
        self.kernel_middle_length = kernel_middle_length
        self.kernel_long_length = kernel_long_length

        self.stride = kernel_short_length // 2

        self.conv_1d_short = nn.Conv1d(1, out_channels, kernel_short_length, stride=self.stride, padding=kernel_short_length - 1)
        self.short_relu = nn.ReLU()

        self.conv_1d_middle = nn.Conv1d(1, out_channels, kernel_middle_length, stride=self.stride, padding=0)
        self.middle_relu = nn.ReLU()

        self.conv_1d_long = nn.Conv1d(1, out_channels, kernel_long_length, stride=self.stride, padding=0)
        self.long_relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        X1 = self.short_relu(self.conv_1d_short(x))

        time_length_short = X1.shape[-1]
        time_length_x = x.shape[-1]
        time_length_middle = (time_length_short - 1) * self.stride + self.kernel_middle_length
        time_length_long = (time_length_short - 1) * self.stride + self.kernel_long_length
        
        middle_padded = nn.functional.pad(x, (0, time_length_middle - time_length_x), 'constant', 0)
        X2 = self.middle_relu(self.conv_1d_middle(middle_padded))

        long_padded = nn.functional.pad(x, (0, time_length_long - time_length_x), 'constant', 0)
        X3 = self.long_relu(self.conv_1d_long(long_padded))

        X = torch.cat([X1, X2, X3], 1)
        return X, X1, X2, X3
    
    def transform_input_lengths(self, lengths):
        return (lengths - self.kernel_short_length) // self.stride + 1


class StackedTCNsBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            speaker_embedding_dim: int,
            hidden_dim: int=256,
            n_stacked_tcns: int=3
        ) -> None:
        super().__init__()
        self.channel_norm = ChannelLayerNorm(in_channels)
        self.conv_1d = nn.Conv1d(in_channels, hidden_dim, 1)
        self.stacked_tcns = nn.ModuleList([
            StackedTCNS(hidden_dim, speaker_embedding_dim)
            for i in range(n_stacked_tcns)
        ])

    def forward(self, Y: torch.Tensor, v: torch.Tensor):
        Y = self.channel_norm(Y)
        Y = self.conv_1d(Y)
        for module in self.stacked_tcns:
            Y = module(Y, v)
        return Y


class StackedTCNS(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            speaker_embedding_dim: int, 
            n_blocks: int=4
        ) -> None:
        super().__init__()
        self.first_tcn = FirstTCNBlock(
            speaker_embedding_dim=speaker_embedding_dim, 
            in_channels=in_channels
        )

        self.other_tcns = nn.Sequential(*[
            TCNBlock(
                in_channels=in_channels, 
                dilation=2 ** i
            )
            for i in range(1, n_blocks)
        ])

    def forward(self, Y: torch.Tensor, v: torch.Tensor):
        y = self.first_tcn(Y, v)
        y = self.other_tcns(y)
        return y


class DecodersBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_dim: int,
            stride: int,
            kernel_short_length: int,
            kernel_middle_length: int,
            kernel_long_length: int
        ) -> None:
        super().__init__()
        self.decoder_short = DecoderBlock(
            in_channels,
            hidden_dim,
            kernel_short_length,
            stride
        )
        self.decoder_middle = DecoderBlock(
            in_channels,
            hidden_dim,
            kernel_middle_length,
            stride
        )
        self.decoder_long = DecoderBlock(
            in_channels,
            hidden_dim,
            kernel_long_length,
            stride
        )

    def forward(self, convolved: torch.Tensor, Y1: torch.Tensor, Y2: torch.Tensor, Y3: torch.Tensor):
        s1 = self.decoder_short(convolved, Y1)
        s2 = self.decoder_middle(convolved, Y2)
        s3 = self.decoder_long(convolved, Y3)
        return s1, s2, s3


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_dim: int,
            kernel_size: int,
            stride: int
        ) -> None:
        super().__init__()
        self.conv_1d = nn.Conv1d(in_channels, hidden_dim, 1)
        self.relu = nn.ReLU()
        self.decoder = nn.ConvTranspose1d(hidden_dim, 1, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor, Y: torch.Tensor):
        M = self.relu(self.conv_1d(x))
        S = self.decoder(Y * M)
        return S.squeeze()


class SpeakerEmbeddingBlock(nn.Module):
    def __init__(
            self, 
            n_classes: int,
            in_channels: int,
            embedding_dim: int=256
        ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings_layer = nn.Sequential(
            ChannelLayerNorm(in_channels),
            nn.Conv1d(in_channels, 128, 1),
            ResNetBlock(128, 128),
            ResNetBlock(128, 256),
            nn.Conv1d(256, embedding_dim, 1),
        )
        self.dense = nn.Linear(embedding_dim, n_classes)

    def forward(self, X: torch.Tensor, ref_length: torch.Tensor):
        embeddings = self.embeddings_layer(X)
        embeddings = embeddings.sum(-1) / ref_length.view(-1, 1).float()

        speaker_preds = self.dense(embeddings)
        return speaker_preds, embeddings