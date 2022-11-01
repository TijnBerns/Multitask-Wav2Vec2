
from torchmetrics import SumMetric
import torch
from typing import Optional, Any, List, Tuple, Union, Dict
from worderrorrate import WER
import numpy as np
from config import Config
from jiwer import wer as jiwer_wer
from pyllr.pav_rocch import PAV, ROCCH

################################################################################
# helper methods for both measures

def _verify_correct_scores(
    groundtruth_scores: List[int], predicted_scores: List[float]
):
    if len(groundtruth_scores) != len(predicted_scores):
        raise ValueError(
            f"length of input lists should match, while"
            f" groundtruth_scores={len(groundtruth_scores)} and"
            f" predicted_scores={len(predicted_scores)}"
        )
    if not all(np.isin(groundtruth_scores, [0, 1])):
        raise ValueError(
            f"groundtruth values should be either 0 and 1, while "
            f"they are actually one of {np.unique(groundtruth_scores)}"
        )
        

################################################################################
# EER (equal-error-rate)

def calculate_eer(
    groundtruth_scores: List[int], predicted_scores: List[float]
) -> float:
    """
    Calculate the equal error rate between a list of groundtruth pos/neg scores
    and a list of predicted pos/neg scores.
    Positive ground truth scores should be 1, and negative scores should be 0.
    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values
    :return: a float value containing the equal error rate and the corresponding threshold
    """
    _verify_correct_scores(groundtruth_scores, predicted_scores)

    scores = np.asarray(predicted_scores, dtype=float)
    labels = np.asarray(groundtruth_scores, dtype=float)
    rocch = ROCCH(PAV(scores, labels))

    eer = rocch.EER()

    return float(eer)


################################################################################
# Speaker change metrics

def _fnr(fn: SumMetric, spch: SumMetric) -> torch.tensor:   
    """Computes false negative rate only considering speaker changes

    Returns:
        tensor: false negative rate
    """
    if spch.value == 0:
        return torch.tensor(-1)
    return (fn / spch).compute()


def _fpr(fp: SumMetric, words: SumMetric) -> torch.tensor:
    """Compute false positive rate only considering speaker changes

    Returns:
        tensor: false positive rate
    """
    if words.value == 0:
        return torch.tensor(-1)
    return (fp / words).compute()


class SpeakerChangeStats():
    def __init__(self, prefix: str = None) -> None:
        self.i = SumMetric()  # number of speaker change insertions
        self.d = SumMetric()  # number of speaker change deletions
        self.s = SumMetric()  # number of speaker change substitutions
        self.c = SumMetric()  # number of speaker change hits

        self.fp = SumMetric()  # number of speaker change false negatives
        self.fn = SumMetric()  # number of speaker change false positives
        self.spch = SumMetric()  # number of speaker changes
        self.word = SumMetric()  # number of words (non-speaker-change)

        self.error = SumMetric()  # number of errors computed by wer
        self.total = SumMetric()  # number of total computed by wer
        self.samples = SumMetric()  # number of samples

        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = f"{prefix}_"

    def __call__(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:
        self._update_stats(preds, target)

    def compute_and_reset(self) -> Dict[str, float]:
        computed_stats = {
            f"{self.prefix}fnr": _fnr(self.fn, self.spch),
            f"{self.prefix}fpr": _fpr(self.fp, self.word),
            f"{self.prefix}wer": (self.error / self.total).compute(),
            f"{self.prefix}fp": (self.fp / self.samples).compute(),
            f"{self.prefix}fn": (self.fn / self.samples).compute(),
        }
        self.reset()
        return computed_stats

    def _operation_counts(self, pralign: List[str]) -> Tuple[int, int, int, int, int, int]:
        c, i, d, s, fn, fp = [0] * 6

        for idx in range(len(pralign[2])):
            if pralign[0][idx] != Config.speaker_change_symbol and pralign[1][idx] != Config.speaker_change_symbol:
                continue

            if pralign[2][idx] == 'c':
                c += 1
            elif pralign[2][idx] == 'i':
                i += 1
            elif pralign[2][idx] == 'd':
                d += 1
            elif pralign[2][idx] == 's':
                s += 1
                if pralign[0][idx] == Config.speaker_change_symbol:
                    fn += 1
                elif pralign[1][idx] == Config.speaker_change_symbol:
                    fp += 1
                else:
                    raise ValueError
            else:
                raise ValueError

        fn += d
        fp += i

        return c, i, d, s, fn, fp

    def _update_stats(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [target]

        for prd, tgt in zip(preds, target):
            # Compute operation counts
            tgt, prd = tgt.split(), prd.split()
            wer = WER(tgt, prd)
            c, i, d, s, fn, fp = self._operation_counts(wer.pralign())

            # Update stats
            self.i.update(i)
            self.d.update(d)
            self.s.update(s)
            self.c.update(c)

            self.fp.update(fp)
            self.fn.update(fn)
            self.spch.update(s + d + c)
            self.word.update(len(tgt) - s - d - c)

            self.error.update(wer.nerr)
            self.total.update(len(tgt))
            self.samples.update(1)
        return

    def reset(self) -> None:
        self.i.reset()
        self.d.reset()
        self.s.reset()
        self.c.reset()

        self.fp.reset()
        self.fn.reset()
        self.spch.reset()
        self.word.reset()

        self.error.reset()
        self.total.reset()
        self.samples.reset()
