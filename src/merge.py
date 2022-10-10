
from pathlib import Path
import torchaudio
import torch
import random
from tqdm import tqdm
from config import Config
import data
from typing import List

wav_idx = 0
srate_idx = 1
trans = 2
speaker_idx = 3
book_idx = 4
snr_idx = 5


def save_pairs(save_loc: Path, merged_pairs: List):
    """Saves pairs of samples in the same format as the original LibriSpeech dataset
    """
    if not save_loc.exists():
        save_loc.mkdir()

    for waveform, sample_rate, transcription, id1, id2, id3 in merged_pairs:
        sample_path = (save_loc / id1 / id2)
        sample_path.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            sample_path / f'{id1}-{id2}-{id3}.flac', waveform, sample_rate)

        write_trans_file(save_loc, id1, id2, id3, transcription)


def write_trans_file(root: Path, id1: str, id2: str, id3: str, transcription: str):
    """Writes the transcription to the transcription file
    """
    trans_file = (root / id1 / id2 / f'{id1}-{id2}.trans.txt')

    if not trans_file.exists():
        trans_file.touch()

    with trans_file.open('a', encoding='utf-8') as f:
        f.write(f'{id1}-{id2}-{id3} {transcription}\n')

    return


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


def main():
    root = Path(Config.datapath)

    tmp_train_set = torchaudio.datasets.LIBRISPEECH(
        root, url="train-clean-100", download=True)
    train_set, val_set = data.split_dataset(tmp_train_set, 0.7)
    test_set = torchaudio.datasets.LIBRISPEECH(
        root, url="test-clean", download=True)
    dev_set = torchaudio.datasets.LIBRISPEECH(
        root, url="dev-clean", download=True)

    for dataset, dataset_str, num_samples in [
        (dev_set, "dev-clean", 10e6),
        (test_set, "test-clean", 10e6),
        (val_set, "val-clean", 10e6),
        (train_set, "train-clean", 10e6)
    ]:

        print(f"Merging samples of {dataset_str}...")
        pair_generator_repeat = PairGeneratorRepeat(num_samples=num_samples,
                                                    min_tokens=Config.min_tokens,
                                                    max_tokens=Config.max_tokens,
                                                    max_attempts=Config.max_attempts)
        merged_pairs = pair_generator_repeat.gen_pairs(dataset, )
        save_pairs(root / (dataset_str + "-rep"), merged_pairs)
        print(
            f"Generated {len(merged_pairs)} samples with repeating speakers for {dataset_str}.\n")

        pair_generator_no_repeat = PairGeneratorNoRepeat(num_samples=num_samples,
                                                         min_tokens=Config.min_tokens,
                                                         max_tokens=Config.max_tokens,
                                                         max_attempts=Config.max_attempts)
        merged_pairs = pair_generator_no_repeat.gen_pairs(dataset)
        save_pairs(root / (dataset_str + "-no-rep"), merged_pairs)
        print(
            f"Generated {len(merged_pairs)} samples with no repeating speakers for {dataset_str}.\n")


if __name__ == "__main__":
    main()
