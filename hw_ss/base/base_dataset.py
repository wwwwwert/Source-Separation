import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_ss.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index
        self.speaker_to_idx = self.get_speaker_to_idx(index)

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        mix_wave = self.load_audio(data_dict["path_mix"])
        # mix_wave, mix_spec = self.process_wave(mix_wave)

        target_wave = self.load_audio(data_dict["path_target"])
        # target_wave, target_spec = self.process_wave(target_wave)

        ref_wave = self.load_audio(data_dict["path_ref"])
        # ref_wave, ref_spec = self.process_wave(ref_wave)

        return {
            "mix_audio": mix_wave,
            # "mix_spec": mix_spec,

            "target_audio": target_wave,
            # "target_spec": target_spec,

            "ref_audio": ref_wave,
            # "ref_spec": ref_spec,

            # "path_mix": data_dict['path_mix'],
            # "path_target": data_dict['path_target'],
            # "path_ref": data_dict['path_ref'],

            "duration": mix_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "speaker_id": self.speaker_to_idx[data_dict["target_id"]],

            # "target_id": data_dict["target_id"],
            # "noise_id": data_dict["noise_id"]
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec

    def _filter_records_from_dataset(
            self, index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] / self.config_parser["preprocessing"]["sr"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)


        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
    
    @staticmethod
    def get_speaker_to_idx(index):
        targets = {entry["target_id"] for entry in index}
        speaker_to_idx = dict(zip(targets, range(len(targets))))
        return speaker_to_idx
    
    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path_mix" in entry, (
                "Each dataset item should include field 'path_mix'"
                " - path to mixed audio."
            )
            assert "path_target" in entry, (
                "Each dataset item should include field 'path_target'"
                " - path to target audio."
            )
            assert "path_ref" in entry, (
                "Each dataset item should include field 'path_ref'"
                " - path to reference audio."
            )
            assert "target_id" in entry, (
                "Each dataset item should include field 'target_id'"
                " - Target speaker's id."
            )
            assert "noise_id" in entry, (
                "Each dataset item should include field 'noise_id'"
                " - Noise speaker's id."
            )
