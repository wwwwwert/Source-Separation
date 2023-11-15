import json
import logging
import os
import random
import shutil
import warnings
from functools import reduce
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_ss.base.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if not index_path.exists():
            self._create_index(part)
        with index_path.open() as f:
            index = json.load(f)
        return index

    def _create_index(self, part):
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        # create and write index
        n_files = sum([len(files) for r, d, files in os.walk(split_dir)])
        test = 'test' in part or 'dev' in part

        path_mixtures = self._data_dir / (part + '_mixtures')

        speakers_ids = [speaker_dir.name for speaker_dir in os.scandir(split_dir)]
        speakers_files = [LibriSpeechSpeakerFiles(speaker_id, split_dir, audioTemplate="*.flac") for speaker_id in speakers_ids]

        mixer = MixtureGenerator(
            speakers_files,
            path_mixtures,
            n_files=n_files,
            test=test
        )
    
        index_df = mixer.generate_mixes(
            snr_levels=[-5, 5],
            num_workers=8,
            update_steps=100,
            trim_db=20,
            vad_db=20,
            audioLen=3
        )

        index_df.to_json(self._data_dir / f"{part}_index.json", orient="records", indent=2)
    
    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))


def snr_mixer(clean, noise, snr):
    """Mix clean audio with noise to get desired SNR
    
    Args:
        clean: array of clean audio
        noise: array of noise audio
        snr: value of desired SNR

    Returns: mixed audio
    """
    # calculating noise from SNR formula
    amp_noise = np.linalg.norm(clean) / 10 ** (snr / 20)

    # scaling noise
    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise

    mix = clean + noise_norm
    return mix

def vad_merge(audio, top_db):
    """Reduce silence via VAD merge.
    
    Args:
        audio: array of audio
        top_db: threshold (in decibels) below reference to consider as silence
    Returns: shrunk audio
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(audio[s:e])
    return np.concatenate(temp, axis=None)

def cut_audios(s1, s2, sec, sr):
    """Cut pair of audios by segment of seconds.

    Args:
        s1: audio sample 1
        s2: audio sample 2
        sec: length in seconds
        sr: sample rate

    Returns: two lists of cut audios
    """
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)

    s1_cut = []
    s2_cut = []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])

        segment += 1

    return s1_cut, s2_cut

def fix_length(s1, s2, mode='max'):
    """
    Args:
        s1: audio sample 1
        s2: audio sample 2
        mode: 'min' or 'max'
    
    Returns:
        pair of cropped audios
    """
    if mode == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    elif mode == 'max':
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    else:
        raise ValueError(f'Unknown mode {mode}')
    return s1, s2

def create_mix(idx, triplet, snr_levels, out_dir, test, sr=16000, **kwargs):
    """Mixes audios in triplet and saves into out_dit.
    Args:
        idx: index of triplet
        triplet: dict of triplet
        snr_levels: list of snr levels
        out_dir, string of output dir
        test: bool whether mixture is for test 
        sr: sample rate for audios
    """
    trim_db, vad_db = kwargs["trim_db"], kwargs["vad_db"]
    audioLen = kwargs["audioLen"]

    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))

    meter = pyln.Meter(sr) # create BS.1770 meter

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    loudsRef = meter.integrated_loudness(ref)

    s1Norm = pyln.normalize.loudness(s1, louds1, -29)
    s2Norm = pyln.normalize.loudness(s2, louds2, -29)
    refNorm = pyln.normalize.loudness(ref, loudsRef, -23.0)

    amp_s1 = np.max(np.abs(s1Norm))
    amp_s2 = np.max(np.abs(s2Norm))
    amp_ref = np.max(np.abs(refNorm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return [[]]

    if trim_db:
        ref, _ = librosa.effects.trim(refNorm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1Norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2Norm, top_db=trim_db)

    if len(ref) < sr:
        return [[]]

    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")

    snr = np.random.choice(snr_levels, 1).item()

    index = [
        # mixed paths
        # target paths
        # ref paths
        # clear id
        # noise id
        # mix length
    ]

    if not test:
        s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audioLen, sr)

        for i in range(len(s1_cut)):
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

            louds1 = meter.integrated_loudness(s1_cut[i])
            s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -23.0)
            loudMix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loudMix, -23.0)

            path_mix_i = path_mix.replace("-mixed.wav", f"_{i}-mixed.wav")
            path_target_i = path_target.replace("-target.wav", f"_{i}-target.wav")
            path_ref_i = path_ref.replace("-ref.wav", f"_{i}-ref.wav")
            sf.write(path_mix_i, mix, sr)
            sf.write(path_target_i, s1_cut[i], sr)
            sf.write(path_ref_i, ref, sr)
            index.append([
                path_mix_i,
                path_target_i,
                path_ref_i,
                target_id,
                noise_id,
                mix.shape[0]
            ])
    else:
        s1, s2 = fix_length(s1, s2, 'max')
        mix = snr_mixer(s1, s2, snr)
        louds1 = meter.integrated_loudness(s1)
        s1 = pyln.normalize.loudness(s1, louds1, -23.0)

        loudMix = meter.integrated_loudness(mix)
        mix = pyln.normalize.loudness(mix, loudMix, -23.0)
        sf.write(path_mix, mix, sr)
        sf.write(path_target, s1, sr)
        sf.write(path_ref, ref, sr)
        index.append([
            path_mix,
            path_target,
            path_ref,
            target_id,
            noise_id,
            mix.shape[0]
        ])
    
    return index

def run_mixer(args):
    idx, triplet, snr_levels, out_dir, test, kwargs = args
    return create_mix(idx, triplet, snr_levels, out_dir, test, **kwargs)


class LibriSpeechSpeakerFiles:
    """Wrapper for LibriSpeech speaker files."""
    def __init__(self, speaker_id: str, audios_dir: str, audioTemplate="*-norm.wav"):
        """Init speaker files.
        Args:
            speaker_id: string of speaker id
            audios_dir: string of dataset directory
            audioTemplate: 
        """
        self.id = speaker_id
        self.audioTemplate=audioTemplate
        self.files = self.find_files_by_worker(audios_dir)

    def find_files_by_worker(self, audios_dir):
        """Finds files of the worker.
        Expects audio files names to end with '*-norm.wav'

        Args:
            audios_dir: string of dataset directory
        """
        speakerDir = os.path.join(audios_dir, self.id)
        chapterDirs = os.scandir(speakerDir)
        files = []
        for chapterDir in chapterDirs:
            files += [
                file
                for file in glob(
                    os.path.join(speakerDir, chapterDir.name, self.audioTemplate)
                )
            ]
        return files


class MixtureGenerator:
    """Class for mixture generation"""
    def __init__(self, speakers_files, out_folder, n_files=5000, test=False, randomState=42):
        """Initialize MixtureGenerator
        Args:
            speaker_files: list of SpeakerFiles for every speaker_id
            out_folder: folder to save mixtures
            n_files: number of files
            test: bool whether mixtures are generated for test
            randomState: random state
        """
        self.speakers_files = speakers_files
        self.out_folder = out_folder
        self.n_files = n_files
        self.test = test
        self.randomState = randomState

        random.seed(self.randomState)
        os.makedirs(self.out_folder, exist_ok=True)

    def generate_triplets(self):
        """Get target, reference and noise files."""
        i = 0
        all_triplets = {
            "reference": [], 
            "target": [], 
            "noise": [], 
            "target_id": [], 
            "noise_id": []
        }
        while i < self.n_files:
            spk1, spk2 = random.sample(self.speakers_files, 2)  # get two speakers

            if len(spk1.files) < 2 or len(spk2.files) < 2:
                continue

            target, reference = random.sample(spk1.files, 2)
            noise = random.choice(spk2.files)
            all_triplets["reference"].append(reference)
            all_triplets["target"].append(target)
            all_triplets["noise"].append(noise)
            all_triplets["target_id"].append(spk1.id)
            all_triplets["noise_id"].append(spk2.id)
            i += 1

        return all_triplets

    def generate_mixes(self, snr_levels=[0], num_workers=8, **mixer_kwargs):
        """Generates mixes with multiple processes.
        
        Args:
            snr_levels: ???
            num_workers: number of pools
            update_steps: number of steps in which statistics are updated
        """
        triplets = self.generate_triplets()

        args = [
            (i, {
                "reference": triplets["reference"][i],
                "target": triplets["target"][i],
                "noise": triplets["noise"][i],
                "target_id": triplets["target_id"][i],
                "noise_id": triplets["noise_id"][i]
            }, snr_levels, self.out_folder, self.test, mixer_kwargs)
            for i in range(self.n_files)
        ]

        warnings.filterwarnings("ignore")
        with Pool(num_workers) as p:
            indices = list(tqdm(p.imap(run_mixer, args), total=len(args)))
        warnings.filterwarnings("default")
        
        index = reduce(lambda x, y: x + y, indices, [])
        index_df = pd.DataFrame(index, columns=['path_mix', 'path_target', 'path_ref', 'target_id', 'noise_id', 'audio_len'])
        index_df = index_df.dropna().reset_index(drop=True)
        return index_df