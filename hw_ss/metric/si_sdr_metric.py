from typing import List

import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SI_SDR


from hw_ss.base.base_metric import BaseMetric


class SI_SDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = SI_SDR()

    def __call__(self, preds: torch.Tensor, target_audio: torch.Tensor, **kwargs):
        # si_sdrs = []
        # for pred, length, target in zip(preds, preds_length, target_audios):
        #     pred = pred[:length]
        #     si_sdrs.append(self.si_sdr(pred, target))
        # return sum(si_sdrs) / len(si_sdrs)
        with torch.no_grad():
            si_sdr = self.si_sdr(preds.cpu(), target_audio.cpu())
        return si_sdr