
from pathlib import Path
from pprint import pprint
import torchaudio
import torch
import random
from tqdm import tqdm
from config import Config
import data


def get_speaker_dict(dataset: torchaudio.datasets.LIBRISPEECH):
    speaker_dict = {}

    for i, sample in tqdm(enumerate(dataset), desc="Generating sample dictionary"):
        if speaker_dict.get(str(sample[3]), -1) == -1:
            speaker_dict[str(sample[3])] = [i]
        else:
            speaker_dict[str(sample[3])].append(i)

    return speaker_dict


def remove_from_dict(speaker_dict: dict, speaker_id: str, sample_idx: int):
    speaker_dict[speaker_id].remove(sample_idx)

    if len(speaker_dict[speaker_id]) == 0:
        speaker_dict.pop(speaker_id)

    return speaker_dict


def gen_pair_no_repeat(dataset: torchaudio.datasets.LIBRISPEECH, speaker_dict: list, max_tokens: int, max_attempts: int):
    current_attempt = 0
    added_tokens = 0
    prev_speaker = -1
    pair = []

    while added_tokens < max_tokens and current_attempt < max_attempts and len(speaker_dict) > 0:
        # Randomly select a speaker
        speaker_id = random.choice(list(speaker_dict.keys()))

        # Check if speaker is same as previously added speaker
        if speaker_id == prev_speaker:
            current_attempt += 1
            continue

        # Add random sample for selected speaker
        sample_idx = random.choice(speaker_dict[speaker_id])
        sample = dataset[sample_idx]
        pair.append(sample)

        # Update the dict and the total number of tokens in the pair
        added_tokens += sample[0].size(dim=1)
        speaker_dict = remove_from_dict(speaker_dict, speaker_id, sample_idx)

    return pair


def gen_pairs(dataset, num_samples: int, max_tokens: int, max_attempts: int):

    speaker_dict = get_speaker_dict(dataset)

    sample_pairs = []
    n = 0
    attempt = 0

    while n < num_samples and attempt < max_attempts and len(speaker_dict) >= 0:
        # Generate a pair of samples
        pair = gen_pair_no_repeat(
            dataset, speaker_dict, max_tokens, max_attempts)

        # Check if a pair has succesfully been generated
        if len(pair) == 0:
            attempt += 1
            continue

        # Append the generated pair
        sample_pairs.append(pair)

        # Update counts
        n += 1
        attempt = 0

    return sample_pairs


def merge_pairs(sample_pairs: list):
    """Merges pairs of samples in a list
    """
    merged_pairs = []
    for pair in sample_pairs:
        pair = list(zip(*pair))

        # Check if all sample rates are 16_000
        assert (all(sample_rate == 16_000 for sample_rate in pair[1]))

        # Construct the sample
        waveform = torch.cat(pair[0], dim=1)
        transcription = merge_transcription(pair[3], pair[2])
        id1 = "&".join(map(str, pair[3]))
        id2 = "&".join(map(str, pair[4]))
        id3 = "&".join([str(id).zfill(4) for id in pair[5]])
        merged_pairs.append([waveform, 16_000, transcription, id1, id2, id3])

    return merged_pairs


def merge_transcription(ids, transcriptions):
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


def main():
    root = Path(Config.datapath)

    tmp_train_set = torchaudio.datasets.LIBRISPEECH(
        root, url="train-clean-100", download=True)
    train_set, val_set = data.split_dataset(tmp_train_set, 0.7)
    test_set = torchaudio.datasets.LIBRISPEECH(
        root, url="test-clean", download=True)
    dev_set = torchaudio.datasets.LIBRISPEECH(
        root, url="dev-clean", download=True)

    for dataset, save_loc, num_samples in [(dev_set, "dev-clean-no-rep", 1000),
                                           (test_set, "test-clean-no-rep", 1000),
                                           (val_set, "val-clean-no-rep", 1000),
                                           (train_set, "train-clean-no-rep", 5000)]:

        print(f"Merging samples of {dataset}...")
        pairs = gen_pairs(dataset=dataset, max_tokens=Config.max_tokens,
                          max_attempts=Config.max_attempts, num_samples=num_samples)
        merged_pairs = merge_pairs(pairs)
        save_pairs(root / save_loc, merged_pairs)


if __name__ == "__main__":
    main()


# def generate_no_repeat(root: Path, dataset: torchaudio.datasets.LIBRISPEECH, speaker_dict: dict, n_speakers: int):
#     """Generates a single sample pair in which two adjacent samples are not from the same speaker
#     """
#     filter(list(root.iterdir()))
#     for child in root.iterdir():
#         if not child.is_dir():
#             continue

#         speaker_id = child.name()

#     # idxs = random.sample(speaker_dict, k=n_speakers)
#     # prev_speaker_id = -1
#     # samples = []

#     # for id in idxs:
#     #     sample = dataset[id]
#     #     if prev_speaker_id == sample[3]:
#     #         return [], []

#     #     prev_speaker_id = sample[3]
#     #     samples.append(sample)

#     # return samples, idxs


# def generate_repeat(dataset: torchaudio.datasets.LIBRISPEECH, speaker_dict: list, n_speakers: int):

#     pass

# def generate_sample_pairs(dataset: torchaudio.datasets.LIBRISPEECH, generate_func, n_pairs: int = 1000,
#                           n_speakers: int = 2, max_attempts: int = 100, kwargs=None):
#     """Generates sample pairs following the provided function
#     """
#     assert (n_pairs * n_speakers < len(dataset))
#     sample_pairs = []
#     n = 0
#     attempt = 0

#     while n < n_pairs and attempt < max_attempts:
#         # Try to generate a pair
#         samples, ids = generate_func(dataset, )

#         # Check if a pair has succesfully been generated
#         if len(samples) != n_speakers:
#             attempt += 1
#             continue

#         # Append the generated pair
#         sample_pairs.append(samples)
#         idxs = list(set(idxs) - set(ids))

#         # Increase counter and set current attempt back to 0
#         n += 1
#         attempt = 0

#     return sample_pairs

# if __name__ == "__main__":
#     random.seed(Config.seed)
#     # # Data downloads
#     # train_ds = torchaudio.datasets.LIBRISPEECH(Config.datapath, url="train-clean-100", download=True)
#     # test_ds = torchaudio.datasets.LIBRISPEECH(Config.datapath, url="test-clean", download=True)
#     dev_ds = torchaudio.datasets.LIBRISPEECH(
#         Config.datapath, url="dev-clean", download=False)

#     sample_pairs = generate_sample_pairs(dev_ds, generate_no_repeat)
#     merged_pairs = merge_pairs(sample_pairs)
#     save_pairs(Config.datapath + "/test", merged_pairs)
