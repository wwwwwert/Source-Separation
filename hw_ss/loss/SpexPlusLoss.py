import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SI_SDR


class SpexPlusLoss(torch.nn.Module):
    def __init__(
            self, 
            alpha: float=0.1, 
            beta: float=0.1,
            gamma: float=0.5
        ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.ce_loss = CrossEntropyLoss()
        self.s1_si_sdr = SI_SDR()
        self.s2_si_sdr = SI_SDR()
        self.s3_si_sdr = SI_SDR()

    def forward(
            self, 
            speaker_preds: torch.Tensor,
            speaker_id: torch.Tensor,
            s1: torch.Tensor,
            s2: torch.Tensor,
            s3: torch.Tensor,
            target_audio: torch.Tensor,
            **batch
        ) -> Tensor:
        ce_loss = self.ce_loss(speaker_preds, speaker_id)
        si_sdr_loss = -(
            (1 - self.alpha - self.beta) * self.s1_si_sdr(s1, target_audio) +
            self.alpha * self.s2_si_sdr(s2, target_audio) +
            self.beta * self.s3_si_sdr(s3, target_audio)
        )
        loss = si_sdr_loss + self.gamma * ce_loss

        return loss
