#!/usr/bin/env python3

import sys
sys.path.append('src')

import pandas as pd
from config import Config
from pathlib import Path



def summary(path):
    df = pd.read_csv(path)
    num_samples = len(df)
    total_utt = 0
    for speaker_id in iter(df["speaker_id"]):
        speaker_id  = str(speaker_id)
        total_utt += speaker_id.count('&') + 1

    print(f"\
        dataset:        {str(path)}\n\
        num. samples:   {num_samples}\n\
        avg. utt.:      {total_utt / num_samples}\n\n\
            ")


if __name__ == "__main__":
    for path in Path(Config.datapath).rglob("*trans.csv"):
        summary(path)
