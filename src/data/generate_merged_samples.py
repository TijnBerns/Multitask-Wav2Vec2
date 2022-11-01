import sys
sys.path.append('src')

import torchaudio
import torch
import random
from tqdm import tqdm
from typing import List
from config import Config


wav_idx = 0
srate_idx = 1
trans = 2
speaker_idx = 3
book_idx = 4
snr_idx = 5


class PairGenerator():
    def __init__(self, min_tokens: int, max_tokens: int, max_attempts: int, num_samples: int) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self. num_samples = num_samples
        pass

    def gen_pairs(self, dataset):
        speaker_dict = self._get_speaker_dict(dataset)
        sample_pairs = []
        n = 0
        attempt = 0

        while n < self.num_samples and attempt < self.max_attempts and len(speaker_dict) >= 0:
            # Generate a pair of samples
            pair = self._gen_pair(dataset, speaker_dict)

            # Check if a pair has succesfully been generated
            if len(pair) == 0:
                attempt += 1
                continue

            # Append the generated pair
            sample_pairs.append(pair)

            # Update counts
            n += 1
            attempt = 0

        merged_pairs = self._merge_pairs(sample_pairs)
        return merged_pairs

    def _gen_pair(self, dataset: torchaudio.datasets.LIBRISPEECH, speaker_dict: list):
        added_tokens = 0
        pair = []
        sample = []

        while added_tokens < self.min_tokens and len(speaker_dict) > 0:
            # Select next sample for given speaker
            sample, sample_idx = self._select_sample(
                dataset, speaker_dict, sample)

            if sample_idx == -1:
                return pair

            # Add sample to the pair and update the number of added tokens
            pair.append(sample)
            speaker_id = str(sample[speaker_idx])
            speaker_dict = self._remove_from_dict(
                speaker_dict, speaker_id, sample_idx)
            added_tokens += sample[wav_idx].size(dim=1)

        return pair

    def _merge_pairs(self, sample_pairs: list):
        """Merges pairs of samples in a list
        """
        merged_pairs = []
        for pair in sample_pairs:

            pair = list(zip(*pair))

            # Check if all sample rates are 16_000
            assert (all(sample_rate == 16_000 for sample_rate in pair[1]))

            # Construct the sample
            waveform = torch.cat(pair[0], dim=1)
            transcription = self._merge_transcription(pair[3], pair[2])
            id1 = "&".join(map(str, pair[3]))
            id2 = "&".join(map(str, pair[4]))
            id3 = "&".join([str(id).zfill(4) for id in pair[5]])
            merged_pairs.append(
                [waveform, 16_000, transcription, id1, id2, id3])

        return merged_pairs

    def _merge_transcription(self, ids, transcriptions):
        assert(len(ids) == len(transcriptions))

        transcription = ""
        for i in range(len(ids) - 1):
            # Diffferent IDs so SC symbol is added
            if ids[i] != ids[i+1]:
                transcription += transcriptions[i] + \
                    f" {Config.speaker_change_symbol} "
            # Same IDs so no SC symbol is added
            else:
                transcription += transcriptions[i] + ' '

        return transcription + transcriptions[-1]

    def _get_speaker_dict(self, dataset: torchaudio.datasets.LIBRISPEECH):
        speaker_dict = {}

        for i, sample in tqdm(enumerate(dataset), desc="Generating sample dictionary"):
            if speaker_dict.get(str(sample[speaker_idx]), -1) == -1:
                speaker_dict[str(sample[speaker_idx])] = [i]
            else:
                speaker_dict[str(sample[speaker_idx])].append(i)

        return speaker_dict

    def _remove_from_dict(self, speaker_dict: dict, speaker_id: str, sample_idx: int):
        speaker_dict[speaker_id].remove(sample_idx)

        if len(speaker_dict[speaker_id]) == 0:
            speaker_dict.pop(speaker_id)

        return speaker_dict

    def _select_sample(self) -> List:
        raise NotImplementedError()


class PairGeneratorNoRepeat(PairGenerator):
    def _select_sample(self, dataset, speaker_dict, prev_sample: List = []):
        speaker_ids = list(speaker_dict.keys())

        if len(prev_sample) != 0:
            if len(speaker_ids) == 1 and speaker_ids[0] == str(prev_sample[speaker_idx]):
                return [], -1
            try:
                speaker_ids.remove(str(prev_sample[speaker_idx]))
            except ValueError:
                pass

        speaker_id = random.choice(speaker_ids)
        sample_idx = random.choice(speaker_dict[speaker_id])
        sample = dataset[sample_idx]
        return sample, sample_idx


class PairGeneratorRepeat(PairGenerator):
    def _select_sample(self, dataset, speaker_dict: dict, prev_sample: list = []):
        speaker_id = list(speaker_dict.keys())[0]
        sample_idx = speaker_dict[speaker_id][0]
        sample = dataset[sample_idx]

        # Check if selected sample is from the same book as the previous sample
        if len(prev_sample) != 0 and sample[book_idx] != prev_sample[book_idx]:
            return [], -1

        return sample, sample_idx
