import utils
from config import Config
from typing import List, Any, Dict
from metrics import SpeakerChangeStats
from pathlib import Path

from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import torch
import pytorch_lightning as pl


def load_processor() -> Wav2Vec2Processor:
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


class Wav2Vec2Module(pl.LightningModule):
    def __init__(self, 
                 num_epochs: int = Config.num_epochs,
                 stage: int = 1,
                 lr_stage_one: float = Config.lr_stage_one,
                 lr_stage_two: float = Config.lr_stage_two,
                 batch_size: int = Config.batch_size,
                 min_lr: float = 0):

        super().__init__()
        self.processor: Wav2Vec2Processor = load_processor()
        self.model: Wav2Vec2ForCTC = load_model()
        self.stage: int = stage
        
        # Metrics
        self.val_stats: SpeakerChangeStats = SpeakerChangeStats(prefix="val")
        self.test_stats: SpeakerChangeStats = SpeakerChangeStats(prefix="test")
        self.test_preds: List[Dict] = []

        # Training parameters
        self.num_epochs: int = num_epochs
        self.lr_stage_one: float = lr_stage_one
        self.lr_stage_two: float = lr_stage_two
        self.min_lr: float = min_lr
        self.batch_size: int = batch_size

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

        self.log("train_loss", output.loss, batch_size=self.batch_size)
        return output.loss

    def validation_step(self, batch, batch_idx):
        waveform = batch['waveform']
        transcription = batch['transcription']
        output, hypothesis = self._train_val_step(waveform, transcription)

        self.val_stats(hypothesis, transcription)
        self.log("val_loss", output.loss, batch_size=self.batch_size)
        return output.loss

    def validation_epoch_end(self, outputs: List[torch.Tensor]) -> None:
        self.log_dict(self.val_stats.compute_and_reset())
        return

    def test_step(self, batch, batch_idx):
        waveform = batch['waveform']
        transcription = batch['transcription']
        output, hypothesis = self._train_val_step(waveform, transcription)

        for trans, hyp in zip(transcription, hypothesis):
            self.test_preds.append({
                "reference": trans,
                "hypothesis": hyp
            })

        self.test_stats(hypothesis, transcription)
        self.log("test_loss", output.loss, batch_size=self.batch_size)
        return output.loss

    def test_epoch_end(self, outputs: Any) -> None:
        self.log_dict(self.test_stats.compute_and_reset())
        return

    def freeze_all_but_head(self) -> None:
        # Freeze all layers except last
        self.model.requires_grad_(False)
        self.model.lm_head.requires_grad_(True)
        return

    def configure_optimizers(self) -> None:
        # setup the optimization algorithm
        if self.stage == 1:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr_stage_one)
        elif self.stage == 2:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr_stage_two)

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
