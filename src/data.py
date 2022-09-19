
import torch
from pathlib import Path
import torchaudio
from config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm


def load_sample(path: Path):
    assert(path.name[-5:] == '.flac')
    ids = path.name[:-5].split('-')
    waveform, sample_rate = torchaudio.load(path)
    transcription = load_transcription(path, ids)
    return (waveform, sample_rate, transcription, *ids)


def load_transcription(path: Path, ids):
    trans_file = path.parent / ("-".join(ids[:-1]) + '.trans.txt')
    speaker_book_id = "-".join(ids)
    transcription = None
    
    with open(trans_file) as f:
        lines = f.readlines()
        for line in lines:
            id, trans = line.split(sep=' ', maxsplit=1)

            if id == speaker_book_id:
                transcription = trans[:-1]
            
    if transcription is None:
        breakpoint()
    return transcription


class CustomLibriSpeechDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        # self.transcription = self._load_transcriptions
        self.samples = self._load_samples(root_dir)

    def _load_samples(self, root_dir):
        file_names = list(Path(root_dir).rglob("*.flac"))
        samples = list(range(len(file_names)))
        for i, file_name in enumerate(file_names):
            samples[i] = load_sample(file_name)
        return samples
    
    # def _load_transcriptions(self, root_dir):
    #     file_names = Path(root_dir).rglob("*.trans.txt")
    #     transcriptions = {}
    #     for trans_file in file_names:
    #         with open(trans_file)as f: 
    #             [line.split(' ', 1) for line in f.readlines()]
    #             transcriptions.update()
            
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate(batch):
    xx, sample_rate, yy, id1, id2, id3 = zip(*batch)
    xx = [x.flatten() for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return {"waveform": xx_pad, "transcription": list(yy), "id1": id1[0], "id2": id2[0], "id3": id3[0]}


def initialize_loader(config: Config, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=pad_collate,
    )

    return dataloader


def get_loaders(config: Config):
    # Initialize datasets and loaders
    # train_ds = torchaudio.datasets.LIBRISPEECH(Config.datapath, url="train-clean-100", download=True)
    # test_ds = torchaudio.datasets.LIBRISPEECH(Config.datapath, url="test-clean", download=True)
    dev_ds = torchaudio.datasets.LIBRISPEECH(
        Config.datapath, url="dev-clean", download=True)

    # n_partitions = 5
    # len_ds = len(train_ds)
    # len_split = len_ds // n_partitions
    # sizes = [len_split] * (n_partitions - 1)
    # sizes.append(len_ds - (n_partitions - 1) * len_split)
    # train_ds = random_split(train_ds, sizes)[version]

    # train_loader = initialize_loader(Config, train_ds)
    # test_loader = initialize_loader(Config, test_ds)
    dev_loader = initialize_loader(Config, dev_ds)

    return dev_loader
