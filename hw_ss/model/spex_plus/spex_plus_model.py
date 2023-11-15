from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DecodersBlock, EncoderBlock, SpeakerEmbeddingBlock, StackedTCNsBlock


class SpexPlus(nn.Module):
    def __init__(
        self,
        n_class: int,
        *args,
        **kwargs
    ):
        super().__init__()
        self.speaker_encoder = EncoderBlock()
        self.embedding = SpeakerEmbeddingBlock(
            n_classes=n_class, 
            in_channels=self.speaker_encoder.out_channels * 3,
        )
        self.stacked_tcns_block = StackedTCNsBlock(
            in_channels=self.speaker_encoder.out_channels * 3,
            speaker_embedding_dim=self.embedding.embedding_dim,
            hidden_dim=self.speaker_encoder.out_channels
        )
        self.decoder = DecodersBlock(
            in_channels=self.speaker_encoder.out_channels,
            hidden_dim=self.speaker_encoder.out_channels,
            stride=self.speaker_encoder.stride,
            kernel_short_length=self.speaker_encoder.kernel_short_length,
            kernel_middle_length=self.speaker_encoder.kernel_middle_length,
            kernel_long_length=self.speaker_encoder.kernel_long_length
        )
        

    def forward(
            self, 
            mix_audio: torch.Tensor, 
            ref_audio: torch.Tensor, 
            ref_length: torch.Tensor,
            **batch
        ):
        mix_time_length = mix_audio.shape[-1]
        mix_audio = mix_audio.unsqueeze(1)
        ref_audio = ref_audio.unsqueeze(1)

        X, _, _, _ = self.speaker_encoder(ref_audio)
        ref_length = self.speaker_encoder.transform_input_lengths(ref_length)
        speaker_preds, v = self.embedding(X, ref_length)

        Y, Y1, Y2, Y3 = self.speaker_encoder(mix_audio)
        convolved = self.stacked_tcns_block(Y, v)
        s1, s2, s3 = self.decoder(convolved, Y1, Y2, Y3)

        output = {
            'preds': s1[:, :mix_time_length], 
            'speaker_preds': speaker_preds,
            's1': s1[:, :mix_time_length], 
            's2': s2[:, :mix_time_length], 
            's3': s3[:, :mix_time_length]
        }
        
        return output

