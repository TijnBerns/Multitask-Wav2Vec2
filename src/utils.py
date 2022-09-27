from config import Config
import re
import os
import torch


def mean_measures(measures: dict, num_batches: int, len_dataset: int):
    'train_hits' 'train_substitutions' 'train_deletions' 'train_insertions'
    for k, v in measures.items():
        if re.search('wer|mer|wil|wip', k) is not None:
            measures[k] = v / num_batches
        if re.search('hits|insertions|deletions|substitutions', k) is not None:
            measures[k] = v / len_dataset

    return measures


def remove_speaker_change_symbol(transcriptions):
    transcriptions = map(lambda x: x.replace(
        Config.speaker_change_symbol, ''), transcriptions)
    return list(transcriptions)


def set_device():
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)
    device = 'cuda' if torch.cuda.is_available and jobid != default_jobid else 'cpu'

    if jobid == default_jobid and device != "cpu":
        exit("Running on GPU without use of slurm!")
    elif jobid != default_jobid and device == "cpu":
        exit("Running slurm job without using GPU!")

    return device, jobid
