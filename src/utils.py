from config import Config
import os
import torch
from typing import List, Dict, Any, Tuple, Union
import json
import pandas as pd
from pathlib import Path
import itertools


def reconstruct_keys(key: str) -> List[str]:
    """ 
    Converts keys of in format 'ls/sp1&sp2/bk1&bk2/utt1&utt2',
    to a list of keys in format 'ls/sp1/bk1/utt1'.

    Example:
        ls/1898&27&6848/145715&124992&76049/0017&0070&0016 
        -> 
        ['ls/1898/145715/0017', 'ls/27/124992/0070', 'ls/6848/76049/0016']
    """
    keys = []
    
    speaker_ids, book_ids, utterance_ids = key.split('/')[1:]
    for k in zip(speaker_ids.split('&'), book_ids.split('&'), utterance_ids.split('&')):
        # TODO: Fix keys when saving transcription files such that there are no leading zeros??
        k = (*k[:-1], str(int(k[-1])))
        keys.append('ls/' + '/'.join(k))
    embedding_idx = get_embedding_idx(speaker_ids)
    return keys, set(embedding_idx)

def get_embedding_idx(speaker_ids):
    # speaker_ids = speaker_ids.split('&')
    # unique_speaker_ids = dict.fromkeys(speaker_ids).keys()
    # speaker_dict = dict(zip(unique_speaker_ids, range(len(unique_speaker_ids))))
    res = []
    prev_id = None
    count = -1
    
    for id in speaker_ids.split('&'):
        if id != prev_id:
            count += 1
        prev_id = id
        res.append(count)
        
    return res
    

def json_dump(path: Union[str, Path], data: Any) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def write_dict_list(path: str, data: List[Dict]) -> None:
    with open(path, 'w') as f:
        f.write(','.join(data[0].keys()) + '\n')
        for entry in data:
            f.write(','.join(entry.values()) + '\n')
    return


def set_device() -> Tuple[str, str]:
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)
    device = 'cuda' if torch.cuda.is_available and jobid != default_jobid else 'cpu'

    if jobid == default_jobid and device != "cpu":
        exit("Running on GPU without use of slurm!")
    elif jobid != default_jobid and device == "cpu":
        exit("Running slurm job without using GPU!")

    return device, jobid


if __name__ == "__main__":
    test_key = "ls/27&27&1898/145715&124992&76049/0017&0070&0016"
    print(reconstruct_keys(test_key))
