
import torch
from pathlib import Path
import torchaudio
from config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LirbriSpeechItem(object):
    def __init__(self, file_name: Path, transcription: str, ids: list):
        self.file_name = file_name
        self.transcription = transcription
        self.ids = ids

    def load_sample(self):
        waveform, sample_rate = torchaudio.load(self.file_name)
        return (waveform, sample_rate, self.transcription, *self.ids)


class CustomLibriSpeechDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.samples = self._load_samples()

    def _load_transcription(self, file_name: Path, ids: list):
        trans_file = file_name.parent / ("-".join(ids[:-1]) + '.trans.txt')
        speaker_book_id = "-".join(ids)
        transcription = None

        with open(trans_file) as f:
            lines = f.readlines()
            for line in lines:
                id, trans = line.split(sep=' ', maxsplit=1)

                if id == speaker_book_id:
                    transcription = trans[:-1]

        return transcription

    def _load_samples(self):
        file_names = list(self.root_dir.rglob("*.flac"))
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


def pad_collate(batch):
    xx, sample_rate, yy, id1, id2, id3 = zip(*batch)
    xx = [x.flatten() for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return {"waveform": xx_pad, "transcription": list(yy), "id1": id1[0], "id2": id2[0], "id3": id3[0]}


def initialize_loader(dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=pad_collate,
    )

    return dataloader
