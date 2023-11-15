from os import listdir
from os.path import isfile, join

from torch_audiomentations import ApplyImpulseResponse
from torch import Tensor

from hw_ss.augmentations.base import AugmentationBase
    

class RoomImpulse(AugmentationBase):
    def __init__(self, sample_rate: int, *args, **kwargs) -> None:
        super().__init__()
        dataset = "hw_ss/augmentations/wave_augmentations/MIT_Survey"
        rir_wavs_paths = [join(dataset, f) for f in listdir(dataset) if isfile(join(dataset, f))]
        self._aug = ApplyImpulseResponse(rir_wavs_paths, p=0.2, sample_rate=sample_rate)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)