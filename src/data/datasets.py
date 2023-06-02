
from pickle import TRUE
import torch
from pathlib import Path
from torch.functional import Tensor
import torchaudio
from config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate
import itertools
from typing import Union, List
import pandas as pd
import torchdata.datapipes as dp


########################################################################################
# Librispeech items and batches


class LirbriSpeechItem(object):
    def __init__(self, file_name: Union[str, Path], transcription: str, speaker_id: str, book_id: str, utterance_id: str):
        self.file_name: Union[str, Path] = file_name
        self.waveform: torch.Tensor = None
        self.sample_rate = None
        self.transcription: str = transcription

        self.key: str = f"ls/{speaker_id}/{book_id}/{utterance_id}"
        self.speaker_id: str = speaker_id
        self.book_id: str = book_id
        self.utterance_id: str = utterance_id

        self.load_sample()

    def load_sample(self):
        self.waveform, self.sample_rate = torchaudio.load(self.file_name)
        return self.waveform

    def __len__(self):
        return self.waveform.shape[-1]

    def __lt__(self, other):
        return len(self) < len(other)


class LirbriSpeechBatch(object):
    def __init__(self,
                 waveforms: torch.Tensor = [],
                 transcriptions: List[str] = [],
                 speaker_ids: List[str] = [],
                 book_ids: List[str] = [],
                 utterance_ids: List[str] = [],
                 keys: List[str] = [],
                 sample_paths: List[str] = []) -> None:
        self.waveforms = waveforms
        self.transcriptions = transcriptions
        self.speaker_ids = speaker_ids
        self.book_ids = book_ids
        self.utterance_ids = utterance_ids
        self.keys = keys
        self.sample_paths = sample_paths

    def to(self, device):
        return LirbriSpeechBatch(
            self.waveforms.to(device),
            self.transcriptions,
            self.speaker_ids,
            self.book_ids,
            self.utterance_ids,
            self.keys,
            self.sample_paths
        )
        
    def __len__(self):
        return len(self.keys)


def row_processor(row: str):
    return LirbriSpeechItem(file_name=row[0], speaker_id=row[1], book_id=row[2], utterance_id=row[3], transcription=row[4])


def pad_collate(batch: Union[List[LirbriSpeechItem], LirbriSpeechItem]):
    if isinstance(batch, LirbriSpeechItem):
        batch = [batch]
        
    return LirbriSpeechBatch(
        waveforms=pad_sequence(
            [sample.waveform.squeeze() for sample in batch], batch_first=True, padding_value=0),
        transcriptions=default_collate(
            [sample.transcription for sample in batch]),
        speaker_ids=default_collate([sample.speaker_id for sample in batch]),
        book_ids=default_collate([sample.book_id for sample in batch]),
        utterance_ids=default_collate(
            [sample.utterance_id for sample in batch]),
        keys=default_collate([sample.key for sample in batch]),
        sample_paths=default_collate([sample.file_name for sample in batch])
    )


def build_datapipe(trans_file: Union[str, List[str]], dynamic_batch_size: bool=False):
    datapipe: dp.iter.IterDataPipe = dp.iter.FileLister(trans_file)
    datapipe = datapipe.open_files(mode='rt')
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(row_processor)
    if dynamic_batch_size:
        datapipe = datapipe.max_token_bucketize(max_token_count=Config.max_token_count,
                                                include_padding=True)
    datapipe = datapipe.collate(collate_fn=pad_collate)
    return datapipe


def check_batch_format(batch: List[any]):
    if len(batch) != 1:
        raise ValueError(f"Unexpted batch shape: got {batch}")
    return batch[0]


def initialize_loader(datapipe, shuffle: bool):
    dataloader = DataLoader(
        dataset=datapipe,
        num_workers=Config.num_workers,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=check_batch_format,
    )
    return dataloader


########################################################################################
# Utility methods LibriSpeech dataset and loader

def split_dataset(dataset, percentage: float):
    assert percentage > 0 and percentage < 1, "Unvalid percentage provided"
    total_count = len(dataset)
    train_count = int(percentage * total_count)
    split_index = _get_split_index(dataset, train_count)

    train_set = torch.utils.data.Subset(dataset, list(range(split_index)))
    val_set = torch.utils.data.Subset(
        dataset, list(range(split_index, len(dataset))))
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (train_count, valid_count))

    return train_set, val_set


def _get_split_index(dataset, start_index):
    split_index = start_index
    speaker_at_split = dataset[start_index][3]
    speaker = speaker_at_split

    while speaker == speaker_at_split:
        speaker = dataset[split_index][3]
        split_index += 1
    return split_index


train_tmp = torchaudio.datasets.LIBRISPEECH(
    root=Config.datapath, url="train-clean-100", download=True)
train_set, val_set = split_dataset(train_tmp, Config.train_split)

clean_datasets = {"train-clean-100": train_set,
                  "val-clean": val_set,
                  "dev-clean": torchaudio.datasets.LIBRISPEECH(root=Config.datapath, url="dev-clean", download=True),
                  "test-clean": torchaudio.datasets.LIBRISPEECH(root=Config.datapath, url="test-clean", download=True),
                  }

