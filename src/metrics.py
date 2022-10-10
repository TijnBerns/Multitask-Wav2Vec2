from torchmetrics import SumMetric
import torch
from typing import Optional, Any, List, Tuple, Union, Dict
from worderrorrate import WER
import numpy as np
from config import Config
from jiwer import wer as jiwer_wer


def _fnr(fn: int, spch: int) -> float:
    """Computes the average false negative rate only considering speaker changes

    Returns:
        float: average false negative rate
    """
    return fn / max(1., spch)


def _fpr(fp, words, spch) -> float:
    """Compute the average false positive rate only considering speaker changes

    Returns:
        float: average false positive rate
    """
    return fp / max(1., (words - spch))


def _spch_er(s: int, i: int, d: int, spch: int) -> float:
    """Computes the speaker change error

    Returns:
        float: average speaker change error
    """
    return (s + i + d) / max(1, spch)


class SpeakerChangeStats():
    def __init__(self, prefix: str = None) -> None:
        self.fnr = SumMetric()
        self.fpr = SumMetric()
        self.wer = SumMetric()
        self.spch_er = SumMetric()
        self.samples = SumMetric()

        self.stats = [self.fnr, self.fpr, self.wer, self.spch_er, self.samples]

        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = f"{prefix}_"

    def __call__(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:
        self.update(preds, target)

    def compute_and_reset(self) -> Dict[str, float]:
        computed_stats = {
            f"{self.prefix}fnr": (self.fnr / self.samples).compute(),
            f"{self.prefix}fpr": (self.fpr / self.samples).compute(),
            f"{self.prefix}spch_er": (self.spch_er / self.samples).compute(),
            f"{self.prefix}wer": (self.wer / self.samples).compute(),
        }
        self.reset()
        return computed_stats

    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:
        self._update_stats(preds, target)

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
            wer = WER(tgt.split(), prd.split())
            pralign = np.array(wer.pralign())
            c, i, d, s, fn, fp = self._operation_counts(pralign)
            spch = s + d + c
            words = len(pralign[0])

            # Update stats
            self.fnr.update(_fnr(fn=fn, spch=spch))
            self.fpr.update(_fpr(fp=fp, words=words, spch=spch))
            self.wer.update(wer.wer())
            self.spch_er.update(_spch_er(s=s, i=i, d=d, spch=spch))
            self.samples.update(1)
        return

    def reset(self) -> None:
        for stat in self.stats:
            stat.reset()
        return
