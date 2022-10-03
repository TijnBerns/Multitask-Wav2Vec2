from config import Config
import re
import os
import torch
from worderrorrate import WER
import numpy as np
import math

# def mean_measures(measures: dict, num_batches: int, len_dataset: int):
#     'train_hits' 'train_substitutions' 'train_deletions' 'train_insertions'
#     for k, v in measures.items():
#         if re.search('wer|mer|wil|wip', k) is not None:
#             measures[k] = v / num_batches
#         if re.search('hits|errors|insertions|deletions|substitutions', k) is not None:
#             measures[k] = v / len_dataset

#     return measures


def mean_measures(measures: dict, num_batches: int, len_dataset: int):
    for k, v in measures.items():
        measures[k] = v / len_dataset

    return measures


def spch_measure(reference, hypothesis):
    wer = WER(reference, hypothesis)
    pralign = np.array(wer.pralign())
    idx = np.where(np.array(pralign[:2]) == Config.speaker_change_symbol)[1]
    idx = np.unique(idx)
    operation, counts = np.unique(pralign[2][idx], return_counts=True)

    ops = {'c': 0, 'i': 0, 'd': 0, 's': 0}
    for op, cnt in zip(operation, counts):
        ops[op] = cnt

    n_spch = max(ops['s'] + ops['d'] + ops['c'], 1)
    spch_er = (ops['i'] + ops['d'] + ops['s']) / n_spch

    measures = {
        "wer": wer.wer(),
        "insertions": wer.nins,
        "deletions": wer.ndel,
        "substitutions": wer.nsub,
        "insertion_rate": wer.nins / len(wer.ref),
        "deletion_rate": wer.ndel / len(wer.ref),
        "substitutions_rate": wer.nsub / len(wer.ref),
        "spch_error": spch_er,
        "spch_insertions": ops['i'],
        "spch_deletions": ops['d'],
        "spch_substiutions:": ops['s'],
        "spch_insertion_rate": ops['i'] / n_spch,
        "spch_deletion_rate": ops['d'] / n_spch,
        "spch_substiution_rate": ops['s'] / n_spch,
    }

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


if __name__ == "__main__":
    gt = "test what"
    hyp = "test asdasdasd maskdj"
    wer = spch_measure(gt, hyp)
