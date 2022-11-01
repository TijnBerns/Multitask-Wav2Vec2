from config import Config
import re
from typing import List
from dataclasses import dataclass
import torch
import sys
sys.path.append('src')


class PreProcessor():
    def __call__(self, waveform):
        raise NotImplementedError("Call to abstract class")


class PostProcessor():
    def __call__(self, transcription: List[str]):
        raise NotImplementedError("Call to abstract class")


class StripSpeakerChange(PostProcessor):
    def __call__(self, transcription: List[str]):
        for i, trans in enumerate(transcription):
            trans = trans.replace('#', ' # ')
            transcription[i] = re.sub(r'\s+', ' ', trans.strip())
        return transcription


class RemoveSpeakerChange(PostProcessor):
    def __call__(self, transcription: List[str]):
        for i, trans in enumerate(transcription):
            transcription[i] = re.sub(r'#+', '', trans.strip())
        return transcription
