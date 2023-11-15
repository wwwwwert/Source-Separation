import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SI_SDR


class SI_SDRLossWrapper(SI_SDR):
    def forward(self, preds, target_audio, **batch) -> Tensor:
        return super().forward(
            preds,
            target_audio
        )
