
from pathlib import Path
from pprint import pprint
import torchaudio
import torch
import random
from tqdm import tqdm
from config import Config
import data


def save_pairs(save_loc: Path, merged_pairs: list):
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
    def gen_pairs(self, dataset, num_samples: int, max_tokens: int, max_attempts: int):
        speaker_dict = self._get_speaker_dict(dataset)
        sample_pairs = []
        n = 0
        attempt = 0

        while n < num_samples and attempt < max_attempts and len(speaker_dict) >= 0:
            # Generate a pair of samples
            pair = self._gen_pair(dataset, speaker_dict, max_tokens)

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

    def _gen_pair(self, dataset: torchaudio.datasets.LIBRISPEECH, speaker_dict: list, max_tokens: int):
        added_tokens = 0
        pair = []
        sample = []

        while added_tokens < max_tokens and len(speaker_dict) > 0:
            # Select next sample for given speaker
            sample, sample_idx = self._select_sample(
                dataset, speaker_dict, sample)

            if sample_idx == -1:
                return pair

            # Add sample to the pair and update the number of added tokens
            pair.append(sample)
            speaker_id = str(sample[3])
            speaker_dict = self._remove_from_dict(
                speaker_dict, speaker_id, sample_idx)
            added_tokens += sample[0].size(dim=1)

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
            if speaker_dict.get(str(sample[3]), -1) == -1:
                speaker_dict[str(sample[3])] = [i]
            else:
                speaker_dict[str(sample[3])].append(i)

        return speaker_dict

    def _remove_from_dict(self, speaker_dict: dict, speaker_id: str, sample_idx: int):
        speaker_dict[speaker_id].remove(sample_idx)

        if len(speaker_dict[speaker_id]) == 0:
            speaker_dict.pop(speaker_id)

        return speaker_dict


class PairGeneratorNoRepeat(PairGenerator):
    def _select_sample(self, dataset, speaker_dict, prev_sample: list = []):
        speaker_ids = list(speaker_dict.keys())

        if len(prev_sample) != 0:
            if len(speaker_ids) == 1 and speaker_ids[0] == str(prev_sample[3]):
                return [], -1
            try:
                speaker_ids.remove(str(prev_sample[3]))
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
        if len(prev_sample) != 0 and sample[4] != prev_sample[4]:
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

    for dataset, dataset_str, num_samples in [(dev_set, "dev-clean", 1000),
                                              (test_set, "test-clean", 1000),
                                              (val_set, "val-clean", 1000),
                                              (train_set, "train-clean", 5000)]:

        print(f"Merging samples of {dataset_str}...")
        pair_generator_repeat = PairGeneratorRepeat()
        merged_pairs = pair_generator_repeat.gen_pairs(dataset, num_samples=num_samples,
                                                       max_tokens=Config.max_tokens, max_attempts=Config.max_attempts)
        save_pairs(root / (dataset_str + "-rep"), merged_pairs)

        pair_generator_no_repeat = PairGeneratorNoRepeat()
        merged_pairs = pair_generator_no_repeat.gen_pairs(dataset, num_samples=num_samples,
                                                          max_tokens=Config.max_tokens, max_attempts=Config.max_attempts)
        save_pairs(root / (dataset_str + "-no-rep"), merged_pairs)


if __name__ == "__main__":
    main()
