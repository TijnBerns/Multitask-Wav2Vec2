from jiwer import transformations
from config import Config
import re


# {'train_wer': 0.07115246364854165, 'train_mer': 0.07092134875414643, 'train_wil': 0.1136031092252225, 'train_wip':
def mean_measures(measures: dict, num_batches: int):
    'train_hits' 'train_substitutions' 'train_deletions' 'train_insertions'
    for k, v in measures.items():
        if re.search('wer|mer|wil|wip', k) is not None:
            measures[k] = v / num_batches
            
    return measures

def remove_speaker_change_symbol(transcriptions):
    transcriptions = map(lambda x: x.replace(Config.speaker_change_symbol, ''), transcriptions)
    return list(transcriptions)