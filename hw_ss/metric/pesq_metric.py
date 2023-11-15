from typing import List

import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

from hw_ss.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, sr: int=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = sr
        self.pesq = PESQ(self.sr, 'wb', n_processes=5)

    def __call__(self, preds: torch.Tensor, target_audio: torch.Tensor, **kwargs):
        with torch.no_grad():
            pesq = self.pesq(preds.cpu(), target_audio.cpu())
        return pesq 