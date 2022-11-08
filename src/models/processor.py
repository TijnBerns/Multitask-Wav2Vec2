from config import Config
import re
from typing import List, Union
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
    def __call__(self, transcription: List[str]) -> List[str]:
        res = []
        for trans in transcription:
            trans = trans.replace('#', ' # ')
            res.append(re.sub(r'\s+', ' ', trans.strip()))
        return res


class RemoveSpeakerChange(PostProcessor):
    def __call__(self, transcription: List[str]) -> List[str]:
        res = []
        for trans in transcription:
            res.append(re.sub(r'#+', '', trans.strip()))
        return res
