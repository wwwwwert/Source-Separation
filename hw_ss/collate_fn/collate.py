import logging
from typing import List

from torch import int32, long, zeros

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    # mix_spec = zeros(
    #     len(dataset_items),
    #     dataset_items[0]['mix_spec'].shape[1],
    #     get_max_length(dataset_items, 'mix_spec')
    # )

    # ref_spec = zeros(
    #     len(dataset_items),
    #     dataset_items[0]['ref_spec'].shape[1],
    #     get_max_length(dataset_items, 'ref_spec')
    # )

    target_audio = zeros(
        len(dataset_items),
        get_max_length(dataset_items, 'target_audio')
    )

    mix_audio = zeros(
        len(dataset_items),
        get_max_length(dataset_items, 'mix_audio')
    )
    ref_audio = zeros(
        len(dataset_items),
        get_max_length(dataset_items, 'mix_audio')
    )
    speaker_id = zeros(len(dataset_items), dtype=long)
    audio_length = zeros(len(dataset_items), dtype=int32)
    ref_length = zeros(len(dataset_items), dtype=int32)

    max_ref_length = ref_audio.shape[1]

    for idx, item in enumerate(dataset_items):
        item_mix_audio = item['mix_audio']
        item_target_audio = item['target_audio']
        item_ref_audio = item['ref_audio'][..., :max_ref_length]

        item_speaker_id = item['speaker_id']
        item_audio_length = item_mix_audio.shape[-1]

        mix_audio[idx, :item_audio_length] = item_mix_audio
        target_audio[idx, :item_audio_length] = item_target_audio
        ref_audio[idx, :item_ref_audio.shape[-1]] = item_ref_audio

        speaker_id[idx] = item_speaker_id
        audio_length[idx] = item_audio_length
        ref_length[idx] = item_ref_audio.shape[-1]

    return {
        'mix_audio': mix_audio,
        'target_audio': target_audio,
        'ref_audio': ref_audio,
        'speaker_id': speaker_id,
        'audio_length': audio_length,
        'ref_length': ref_length
    }

def get_max_length(dataset_items, element):
    return max([item[element].shape[-1] for item in dataset_items])