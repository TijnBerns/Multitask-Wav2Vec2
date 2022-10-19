from pathlib import Path
from tqdm import tqdm
from typing import List
import torchaudio
import json
from config import Config
from data.datasets import CustomLibriSpeechDataset
from typing import List, Union, Dict
import pandas as pd
import re
from torch.utils.data.dataset import Subset


wav_idx = 0
srate_idx = 1
trans_idx = 2
speaker_idx = 3
book_idx = 4
ut_idx = 5
sample_path_idx = 6

vocab_base = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
              "|": 4, "E": 5, "T": 6, "A": 7, "O": 8,
              "N": 9, "I": 10, "H": 11, "S": 12, "R": 13,
              "D": 14, "L": 15, "U": 16, "M": 17, "W": 18,
              "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23,
              "B": 24, "V": 25, "K": 26, "'": 27, "X": 28,
              "J": 29, "Q": 30, "Z": 31}


def save_pairs(root: Path, merged_pairs: List) -> None:
    """Saves pairs of samples in the same format as the original LibriSpeech dataset
    """
    if not root.exists():
        root.mkdir()

    for waveform, sample_rate, transcription, id1, id2, id3 in merged_pairs:
        sample_path = (root / id1 / id2)
        sample_path.mkdir(parents=True, exist_ok=True)
        sample_path = sample_path / f'{id1}-{id2}-{id3}.flac'
        torchaudio.save(sample_path, waveform, sample_rate)
        write_trans(root, sample_path, id1, id2, id3, transcription)


# def write_trans_clean(root: Path) -> None:
#     """Write a transcription file similar to the custom dataset transcriptions
#     """
#     trans_files = list(root.rglob("*.trans.txt"))
#     for trans_file in trans_files:
#         with trans_file.open("r") as f:
#             for line in f.readlines():
#                 ids, trans = line[:-1].split(maxsplit=1)
#                 id1, id2, id3 = ids.split('-')
#                 sample_path = trans_file.parent / f'{id1}-{id2}-{id3}.flac'
#                 write_trans(root, sample_path,  id1, id2, id3, trans)

def write_trans_clean(dataset, dataset_str: str, target_trans: str):
    if isinstance(dataset, Subset):
        root = Path(dataset.dataset._path)
    else:
        root = Path(dataset._path)
        
    for sample in tqdm(dataset, f"writing transcription for {dataset_str}"):
        id1 = str(sample[speaker_idx])
        id2 = str(sample[book_idx])
        id3 = str(sample[ut_idx])
        trans = sample[trans_idx]
        sample_path = root / id1 / id2 / f"{id1}-{id2}-{id3:0>4}.flac"
        write_trans(Path(target_trans), sample_path,  id1, id2, id3, trans, prefix=dataset_str)


def write_trans(root: Path, sample_path: Path, id1: str, id2: str, id3: str,
                transcription: str, prefix: str = None) -> None:
    """Writes the transcription to the transcription file

    Args:
        root (Path): Folder in which the transcription csv will be written.
        sample_path (Path): Path to the audio file corresponding to provided transcription
        id1 (str): Speaker id.
        id2 (str): Book id.
        id3 (str): Utterance id.
        transcription (str): The transcription.
        prefix (str, optional): If provided, add prefix to saved csv file. By default file is saved as trans.csv.
    """
    if prefix is None:
        trans_file = root / f"trans.csv"
    else:
        trans_file = root / f"{prefix}.trans.csv"

    if not trans_file.exists():
        trans_file.touch()
        trans_file.write_text(
            "path,speaker_id,book_id,utterance_id,transcription\n")

    with trans_file.open('a', encoding='utf-8') as f:
        f.write(",".join([str(sample_path), id1, id2,
                id3, f"{transcription}"]) + "\n")

    return


def add_speaker_start(row: pd.Series) -> pd.Series:
    """Adds a speaker change symbol at the beginning of the transcription

    Args:
        row (pd.Series): Row to which the speaker change symbol is added.

    Returns:
        pd.Series: Altered dataframe row
    """
    transcription = row["transcription"]
    row["transcription"] = f"{Config.speaker_change_symbol} {transcription}"
    return row


def add_speaker_ids(row: pd.Series, spch_symbol: bool = False) -> pd.Series:
    """Replaces the speaker change symbol in the transcription with the speaker id 

    Args:
        row (pd.Series): Row of which the transcription is altered.
        spch_symbol (bool, optional): Whether to keep the speaker change symbol. Defaults to False.

    Returns:
        pd.Series: Altered dataframe row
    """
    speaker_ids = row["speaker_id"].split("&")
    for speaker_id in speaker_ids:
        if spch_symbol:
            row["transcription"] = re.sub(
                "#", f"{Config.speaker_change_symbol}{speaker_id}", row["transcription"], count=1)
        else:
            row["transcription"] = re.sub(
                "#", speaker_id, row["transcription"], count=1)
    return row


def write_trans_from_source(source_trans: Path, target_trans: Path, trans_fn: bool = False):
    df = pd.read_csv(source_trans)
    df = df.astype({"speaker_id": str}, errors='raise')
    df = df.apply(trans_fn, axis=1)
    df.to_csv(target_trans, index=False)
    return


def write_speaker_id_vocab(dataset: CustomLibriSpeechDataset,
                           spid_vocab_path: Union[Path, str]) -> None:
    if isinstance(spid_vocab_path, str):
        spid_vocab_path = Path(spid_vocab_path)
    # if isinstance(spch_vocab_path, str):
    #     spch_vocab_path = Path(spch_vocab_path)

    speaker_ids = set()
    for sample in tqdm(dataset, "obtaining unique speakers"):
        speaker_ids.add(sample[speaker_idx])

    speaker_ids = list(speaker_ids)
    speaker_ids.sort()

    spid_vocab = _gen_spid_vocab_dict(speaker_ids)
    if not spid_vocab_path.exists():
        spid_vocab_path.touch
    with open(spid_vocab_path, 'w') as f:
        json.dump(spid_vocab, f, indent=2)

    # spch_vocab = _gen_spch_vocab_dict(len(speaker_ids))
    # if not spch_vocab_path.exists():
    #     spch_vocab_path.touch
    # with open(spch_vocab_path, 'w') as f:
    #     json.dump(spch_vocab, f, indent=2)


def _gen_spid_vocab_dict(speaker_ids: List[Union[str, int]]):
    vocab = dict(vocab_base)
    start_logit = max(vocab.values()) + 1

    vocab["#"] = start_logit
    start_logit += 1

    for i, speaker_id in enumerate(speaker_ids):
        vocab[str(speaker_id)] = start_logit + i
    return vocab


# def _gen_spch_vocab_dict(num_speaker_ids: int):
#     vocab = dict(vocab_base)
#     start_logit = max(vocab.values()) + 1

#     vocab[Config.speaker_change_symbol] = start_logit
#     for i in range(1, num_speaker_ids):
#         vocab[f"<unk-{i}>"] = start_logit + i
#     return vocab
