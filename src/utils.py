from config import Config
import re


def mean_measures(measures: dict, num_batches: int):
    'train_hits' 'train_substitutions' 'train_deletions' 'train_insertions'
    for k, v in measures.items():
        if re.search('wer|mer|wil|wip', k) is not None:
            measures[k] = v / num_batches

    return measures


def remove_speaker_change_symbol(transcriptions):
    transcriptions = map(lambda x: x.replace(
        Config.speaker_change_symbol, ''), transcriptions)
    return list(transcriptions)
