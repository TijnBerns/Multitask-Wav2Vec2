
from pathlib import Path
from pprint import pprint
import torchaudio
import torch
import random
from tqdm import tqdm
from config import Config



def generate_no_repeat(dataset: torchaudio.datasets.LIBRISPEECH, idxss: list, n_speakers: int):
    """Generates a single sample pair in which two adjacent samples are not from the same speaker
    """
    idxs = random.sample(idxss, k=n_speakers)
    prev_speaker_id = -1
    samples = []

    for id in idxs:
        sample = dataset[id]
        if prev_speaker_id == sample[3]:
            return [], []

        prev_speaker_id = sample[3]
        samples.append(sample)

    return samples, idxs


def generate_repeat(dataset: torchaudio.datasets.LIBRISPEECH, idxss: list, n_speakers: int):
    idx = random.choice(idxss)
    samples = dataset[idx]
    

    return samples, idx


def generate_sample_pairs(dataset: torchaudio.datasets.LIBRISPEECH, generate_func, n_pairs: int = 1000,
                          n_speakers: int = 2, max_attempts: int = 100):
    """Generates sample pairs following the provided function
    """
    assert (n_pairs * n_speakers < len(dataset))

    idxs = range(len(dataset))
    sample_pairs = []
    n = 0
    attempt = 0

    while n < n_pairs and attempt < max_attempts:
        # Try to generate a pair
        samples, ids = generate_func(dataset, idxs, n_speakers)

        # Check if a pair has succesfully been generated
        if len(samples) != n_speakers:
            attempt += 1
            continue

        # Append the generated pair
        sample_pairs.append(samples)
        idxs = list(set(idxs) - set(ids))

        # Increase counter and set current attempt back to 0
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
            transcription += transcriptions[i] + Config.speaker_change_symbol
        # Same IDs so no SC symbol is added
        else:
            transcription += transcriptions[i] + ' '

    return transcription + transcriptions[-1]


def save_pairs(root_str: str, merged_pairs: list):
    """Saves pairs of samples in the same format as the original LibriSpeech dataset
    """
    root = Path(root_str)
    if not root.exists():
        root.mkdir()

    for waveform, sample_rate, transcription, id1, id2, id3 in merged_pairs:
        sample_path = (root / id1 / id2)
        sample_path.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            sample_path / f'{id1}-{id2}-{id3}.flac', waveform, sample_rate)

        write_trans_file(root, id1, id2, id3, transcription)


def write_trans_file(root: Path, id1: str, id2: str, id3: str, transcription: str):
    """Writes the transcription to the transcription file
    """
    trans_file = (root / id1 / id2 / f'{id1}-{id2}.trans.txt')

    if not trans_file.exists():
        trans_file.touch()

    with trans_file.open('a', encoding='utf-8') as f:
        f.write(f'{id1}-{id2}-{id3} {transcription}\n')


if __name__ == "__main__":
    random.seed(Config.seed)
    # # Data downloads
    # train_ds = torchaudio.datasets.LIBRISPEECH(Config.datapath, url="train-clean-100", download=True)
    # test_ds = torchaudio.datasets.LIBRISPEECH(Config.datapath, url="test-clean", download=True)
    dev_ds = torchaudio.datasets.LIBRISPEECH(
        Config.datapath, url="dev-clean", download=False)

    sample_pairs = generate_sample_pairs(dev_ds, generate_no_repeat)
    merged_pairs = merge_pairs(sample_pairs)
    save_pairs(Config.datapath + "/test", merged_pairs)
