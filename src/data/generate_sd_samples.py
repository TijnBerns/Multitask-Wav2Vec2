import sys
sys.path.append('src')
import torchaudio
import torch
from typing import Union, List, Dict
from pathlib import Path
from config import Config
from collections import defaultdict
import random


class SDSample():
    def __init__(self, id: str, rttm: str, recording: torch.Tensor) -> None:
        self.id = id
        self.rttm = rttm
        self.recording = recording
        
    def __repr__(self) -> str:
        return f'SDSample("{self.id}, {self.rttm}, {self.recording}")'

class SDSampleGenerator():
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: Dict[str, List[Path]] = self._get_files(root)
        self.speakers = set(self.files.keys())

        self.MIN_SPEAKERS = 2
        self.MAX_SPEAKERS = 5
        self.MAX_DURATION = 16_000 * 60
        self.MAX_ATTEMPTS = 100

    def _get_files(self, root: Path):
        files = {}
        speaker_paths = [x for x in root.glob("**/*") if x.is_dir()]
        for path in speaker_paths:
            files[path.name] = list(path.rglob("*.flac"))
        return files

    def generate_samples(self):
        sample_count = 0
        while True:
            id = f"custom_{sample_count:>010}"
            sd_sample = self._generate_single_sample(id)

            sample_count += 1
            if len(self.speakers) == 0:
                break        

    def _generate_single_sample(self, id):
        # Generate random number of speakers between MIN and MAX and selected that many speakers
        num_speakers = random.randint(self.MIN_SPEAKERS, self.MAX_SPEAKERS)
        num_speakers = min(num_speakers, len(self.speakers))
        selected_speakers = set(random.sample(self.speakers, num_speakers))

        total_duration = 0
        utterances = []
        speakers = []
        attempts = 0
        speaker = set()
        while total_duration < self.MAX_DURATION:
            # Select random speak and utterance
            speaker = random.choice(list(selected_speakers - speaker))
            utterance_file = random.choice(self.files[speaker])
            utterance = torchaudio.load(utterance_file)[0]
            duration = utterance.shape[-1]

            if total_duration + duration > self.MAX_DURATION:
                break            
            
            # Add selected utterance and speaker
            utterances.append(utterance)
            speakers.append(speaker)
            
            # Remove selected utterance from files 
            self.files[speaker].remove(utterance_file)
            if len(self.files[speaker]) == 0:
                self.files.pop(speaker)
                self.speakers.remove(speaker)
                selected_speakers.remove(speaker)
                
            total_duration += duration
            speaker = set([speaker])
            
            if len(list(selected_speakers - speaker)) == 0:
                break
            
        # Generate RTTM string     
        rttm = self._generate_rttm(id, utterances, speakers)
        
        # Concatenate selected utterances
        recording = torch.cat(utterances, dim=1)
        return SDSample(id, rttm, recording)

    def _generate_rttm(self, id, utterances: list, speakers: list):
        rttm = ""
        start = 0
        for utterance, speaker in zip(utterances, speakers):
            duration = utterance.shape[-1] / 16_000
            rttm += f"SPEAKER {id} 1 {start:.6f} {duration:.6f} <NA> <NA> {speaker} <NA> <NA>\n" 
            start += duration
        return rttm


if __name__ == "__main__":
    dev_path = Path(Config.datapath) / "LibriSpeech" / "dev-clean"
    test_path = Path(Config.datapath) / "LibriSpeech" / "test-clean"
    generator = SDSampleGenerator(dev_path)
    generator.generate_samples()
