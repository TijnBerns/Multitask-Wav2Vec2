from torch.utils.data.dataset import DFIterDataPipe
from config import Config
import re
import os
import torch
from worderrorrate import WER
import numpy as np
import math
from typing import List, Dict, Tuple
import json
import pandas as pd


def write_dict_list(path: str, data: List[Dict]) -> None:
    df = pd.DataFrame(data)
    df.to_csv(path)
    return

def remove_speaker_change_symbol(transcriptions: List[str]) -> List[str]:
    transcriptions = map(lambda x: x.replace(
        Config.speaker_change_symbol, ''), transcriptions)
    return list(transcriptions)


def set_device() -> Tuple[str, str]:
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)
    device = 'cuda' if torch.cuda.is_available and jobid != default_jobid else 'cpu'

    if jobid == default_jobid and device != "cpu":
        exit("Running on GPU without use of slurm!")
    elif jobid != default_jobid and device == "cpu":
        exit("Running slurm job without using GPU!")

    return device, jobid
