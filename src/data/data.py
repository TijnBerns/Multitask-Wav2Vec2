
from pickle import TRUE
import torch
from pathlib import Path
import torchaudio
from config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import itertools
from typing import Union, List


class LirbriSpeechItem(object):
    def __init__(self, file_name: Path, transcription: str, ids: list):
        self.file_name = file_name
        self.transcription = transcription
        self.id1 = ids[0]
        self.id2 = ids[1]
        self.id3 = ids[2]

    def load_sample(self):
        waveform, sample_rate = torchaudio.load(self.file_name)
        return (waveform, sample_rate, self.transcription, self.id1, self.id2, self.id3)


class CustomLibriSpeechDataset(Dataset):
    def __init__(self, root_dir: Union[str, List[str]], speaker_change: bool = True) -> None:
        super().__init__()

        if type(root_dir) == str:
            self.root_dir = [Path(root_dir)]
        else:
            self.root_dir = list(map(Path, root_dir))

        self.speaker_change = speaker_change
        self.samples = self._load_samples()

    def _load_transcription(self, file_name: Path, ids: list):
        trans_file = file_name.parent / ("-".join(ids[:-1]) + '.trans.txt')
        speaker_book_id = "-".join(ids)
        transcription = None

        with open(trans_file) as f:
            lines = f.readlines()
            for line in lines:
                id, trans = line.split(sep=' ', maxsplit=1)

                if id != speaker_book_id:
                    continue

                transcription = trans[:-1]  # Removes the newline character
                if not self.speaker_change:
                    transcription = transcription.replace(
                        f"{Config.speaker_change_symbol}", "")

                return transcription

        raise ValueError(
            f"Could not find transcription of {speaker_book_id} in {trans_file}.")

    def _load_samples(self):
        file_names = [list(dir.rglob("*.flac")) for dir in self.root_dir]
        file_names = list(itertools.chain(*file_names))
        samples = [0] * len(file_names)
        for i, file_name in enumerate(file_names):
            ids = file_name.name[:-5].split('-')
            transcription = self._load_transcription(file_name, ids)
            samples[i] = LirbriSpeechItem(file_name, transcription, ids)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].load_sample()


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


def pad_collate(batch):
    xx, sample_rate, yy, id1, id2, id3 = zip(*batch)
    xx = [x.flatten() for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return {"waveform": xx_pad, "transcription": list(yy), "id1": id1, "id2": id2, "id3": id3}


def initialize_loader(dataset, shuffle: bool):
    dataloader = DataLoader(
        dataset,
        num_workers=Config.num_workers,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=pad_collate,
    )
    return dataloader


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