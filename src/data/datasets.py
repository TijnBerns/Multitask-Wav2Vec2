
from pickle import TRUE
import torch
from pathlib import Path
import torchaudio
from config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import itertools
from typing import Union, List
import pandas as pd


class CustomLibriSpeechDataset(Dataset):
    def __init__(self, trans_file: Union[list, str]) -> None:
        super().__init__()

        if type(trans_file) == str:
            self.trans_file = [trans_file]
        elif type(trans_file) == list:
            self.trans_file = trans_file
        else:
            raise ValueError(
                f"Exptected trans_file to be of type None or str but got {type(trans_file)}.")

        self.samples: List[LirbriSpeechItem] = self._load_samples()

    def _load_transcriptions(self):
        transcriptions = pd.concat((pd.read_csv(f) for f in self.trans_file))
        transcriptions.drop_duplicates()
        return transcriptions

    def _load_samples(self):
        transcriptions = self._load_transcriptions()
        samples = [0] * len(transcriptions)
        for i, sample in enumerate(transcriptions.iterrows()):
            sample = sample[1]
            samples[i] = LirbriSpeechItem(file_name=sample["path"],
                                          transcription=sample["transcription"],
                                          speaker_id=sample["speaker_id"],
                                          book_id=sample["book_id"],
                                          utterance_id=sample["utterance_id"])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].load_sample()


class LirbriSpeechItem(object):
    def __init__(self, file_name: Path, transcription: str, speaker_id: str, book_id: str, utterance_id: str):
        self.file_name = file_name
        self.transcription = transcription
        self.speaker_id = speaker_id
        self.book_id = book_id
        self.utterance_id = utterance_id

    def load_sample(self):
        waveform, sample_rate = torchaudio.load(self.file_name)
        return (waveform, sample_rate, self.transcription, self.speaker_id, self.book_id, self.utterance_id, self.file_name)


class CustomLoader(object):
    def __init__(self, dataset: Dataset, max_tokens: int, drop_last: bool, shuffle: bool, collate_fn):
        self.ds = dataset
        self.max_tokens = max_tokens
        self.drop_last = drop_last
        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)
        self.collate_fn = collate_fn

    def __iter__(self):
        batch = []
        added_tokens = 0
        for idx in self.sampler:
            num_tokens = self.ds[idx][0].size()[-1]
            if added_tokens + num_tokens > self.max_tokens:
                yield self.collate_fn(batch)
                added_tokens = 0
                batch = [self.ds[idx]]
            else:
                batch.append(self.ds[idx])
                added_tokens += num_tokens

        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)


def initialize_loader(dataset, shuffle: bool):
    dataloader = DataLoader(
        dataset,
        num_workers=Config.num_workers,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=_pad_collate,
    )
    return dataloader


def _pad_collate(batch):
    xx, sample_rate, yy, id1, id2, id3, sp = zip(*batch)
    xx = [x.flatten() for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return {"waveform": xx_pad, "transcription": list(yy), "id1": id1, "id2": id2, "id3": id3, "sample_path": sp}


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
    root=Config.datapath, url="train-clean-100")
train_set, val_set = split_dataset(train_tmp, Config.train_split)

clean_datasets = {"train-clean-100": train_set,
                  "val-clean": val_set,
                  "dev-clean": torchaudio.datasets.LIBRISPEECH(root=Config.datapath, url="dev-clean"),
                  "test-clean": torchaudio.datasets.LIBRISPEECH(root=Config.datapath, url="test-clean"),
                  }
