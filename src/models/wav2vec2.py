from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.models import wav2vec2
from config import Config
import torch
import pytorch_lightning as pl
from torchmetrics import WordErrorRate, Metric
from typing import List, Any, Optional, Union
import data
import utils
from worderrorrate import WER
import numpy as np


def load_processor() -> Wav2Vec2Processor:
    # Initialize processor
    tokenizer = Wav2Vec2CTCTokenizer(
        "src/models/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=Config.sample_rate, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor


def load_model() -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h", ctc_loss_reduction="mean")
    model.config.vocab_size = 33
    model.lm_head = torch.nn.Linear(in_features=768, out_features=33)
    model.freeze_feature_encoder()
    return model


class SpeakerChangeError(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    errors: torch.Tensor
    total: torch.Tensor

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("errors", torch.tensor(
            0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(
            0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]):
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [target]

        for tgt, prd in zip(target, preds):
            wer = WER(tgt.split(), prd.split())
            pralign = np.array(wer.pralign())
            idx = np.where(np.array(pralign[:2])
                           == Config.speaker_change_symbol)
            idx = np.unique(idx)
            operation, counts = np.unique(pralign[2][idx], return_counts=True)
            ops = {'c': 0, 'i': 0, 'd': 0, 's': 0}
            for op, cnt in zip(operation, counts):
                ops[op] = cnt

            self.total += ops['s'] + ops['d'] + ops['c']
            self.errors += ops['i'] + ops['d'] + ops['s']

    def compute(self):
        return self.errors / max(1, self.total)


class Wav2Vec2Module(pl.LightningModule):
    def __init__(self, num_epochs: int, lr: float, min_lr: float = 0):
        super().__init__()
        self.processor = load_processor()
        self.model = load_model()

        self.train_wer = WordErrorRate()
        self.train_sper = SpeakerChangeError()
        self.val_wer = WordErrorRate()
        self.val_sper = SpeakerChangeError()

        self.train_measures = []
        self.test_measures = []
        self.val_measures = []

        # Training parameters
        self.num_epochs = num_epochs
        self.lr = lr
        self.min_lr = min_lr

    def _train_val_step(self, waveform: torch.Tensor, transcription: List[str]):
        # Retrieve input values
        input_values = self.processor(
            waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
        input_values = torch.reshape(
            input_values, (len(waveform), -1)).to(self.device)

        # Retrieve target labels
        with self.processor.as_target_processor():
            labels = self.processor(transcription, padding=True,
                                    return_tensors='pt').input_ids
            labels = labels.to(self.device)

        # Compute loss by passing labels
        output = self.model(input_values, labels=labels)

        # Compute and evaluate hypothesis
        logits = output.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        hypothesis = self.processor.batch_decode(predicted_ids)

        return output, hypothesis

    def training_step(self, batch, batch_idx):
        waveform = batch['waveform']
        transcription = batch['transcription']

        output, hypothesis = self._train_val_step(waveform, transcription)
        self.train_wer(hypothesis, transcription)
        self.train_sper(hypothesis, transcription)
        self.log("train_loss", output.loss)
        return output.loss

    def training_epoch_end(self, outputs: List[torch.Tensor]) -> None:
        self.log("train_wer", self.train_wer)
        self.log("train_spcher", self.train_sper)

    def validation_step(self, batch, batch_idx):
        waveform = batch['waveform']
        transcription = batch['transcription']
        output, hypothesis = self._train_val_step(waveform, transcription)

        self.val_wer(hypothesis, transcription)
        self.val_sper(hypothesis, transcription)
        self.log("val_loss", output.loss)
        return output.loss

    def training_epoch_end(self, outputs: List[torch.Tensor]) -> None:
        self.log("val_wer", self.val_wer)
        self.log("val_spcher", self.val_sper)

    def test_step(self, batch, batch_idx):
        waveform = batch['waveform']
        transcription = batch['transcription']
        output, hypothesis = self._train_val_step(waveform, transcription)
        self.log("test_loss", output.loss)
        return output.loss

    def freeze_all_but_head(self) -> None:
        # Freeze all layers except last
        self.model.requires_grad_(False)
        self.model.lm_head.requires_grad_(True)

    def configure_optimizers(self):
        # setup the optimization algorithm
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # setup the learning rate schedule.
        schedule = {
            # Required: the scheduler instance.
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=self.min_lr
            ),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after an optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return [optimizer], [schedule]
