import logging
from os import path
from pathlib import Path

from hw_ss.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, *args, **kwargs):
        index = []
        index = [
            # mixed paths
            # target paths
            # ref paths
            # clear id
            # noise id
            # mix length
            
        ]
        audio_dir = Path(audio_dir)
        for path_mix in (audio_dir / 'mix').iterdir():
            entry = {}
            suffix = path_mix.suffix
            if suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                prefix, _ = path.splitext(str(path_mix.name))
                prefix = prefix[:-5]
                
                entry["path_mix"] = str(path_mix)
                entry["path_target"] = str(audio_dir / 'targets' / (prefix + 'target' + suffix))
                entry["path_ref"] = str(audio_dir / 'mix' / (prefix + 'mixed' + suffix))
                entry["target_id"] = -1
                entry["noise_id"] = -1
            index.append(entry)
        super().__init__(index, *args, **kwargs)
