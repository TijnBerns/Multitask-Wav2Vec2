from config import Config
from typing import List, Any, Dict, Tuple, Union, Optional
from evaluation.metrics import SpeakerChangeStats

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from transformers.modeling_outputs import CausalLMOutput
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from models.processor import StripSpeakerChange, PostProcessor


def load_processor(vocab_path: Union[str, None] = None,) -> Wav2Vec2Processor:
    if vocab_path is None:
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base")
        vocab_size = 32
    else:
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=Config.sample_rate, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer)
        vocab_size = tokenizer.vocab_size

    return processor, vocab_size


def load_model(vocab_size: int) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", ctc_loss_reduction="mean")
    model.config.vocab_size = vocab_size
    model.lm_head = torch.nn.Linear(in_features=768, out_features=vocab_size)
    model.freeze_feature_encoder()
    return model


class Wav2Vec2Module(pl.LightningModule):
    def __init__(self,
                 vocab_path: str = None,
                 num_epochs: int = Config.num_epochs,
                 stage: int = 1,
                 lr_stage_one: float = Config.lr_stage_one,
                 lr_stage_two: float = Config.lr_stage_two,
                 batch_size: int = Config.batch_size,
                 min_lr: float = 0,
                 save_embeddings: bool = False,
                 kernel_size: Optional[int] = None, 
                 postprocessor: PostProcessor = StripSpeakerChange()):

        super().__init__()
        self.processor, self.vocab_size = load_processor(vocab_path)
        self.model: Wav2Vec2ForCTC = load_model(self.vocab_size)
        self.stage: int = stage
        self.postprocessor = postprocessor
        
        # Speaker embeddings
        self.save_embeddings: bool = save_embeddings
        self.embeddings: torch.Tensor = torch.Tensor()
        self.kernel_size: int = kernel_size

        # Metrics
        self.val_stats: SpeakerChangeStats = SpeakerChangeStats(prefix="val")
        self.test_stats: SpeakerChangeStats = SpeakerChangeStats(prefix="dev")
        self.test_preds: List[Dict] = []

        # Training parameters
        self.num_epochs: int = num_epochs
        self.lr_stage_one: float = lr_stage_one
        self.lr_stage_two: float = lr_stage_two
        self.min_lr: float = min_lr
        self.batch_size: int = batch_size

    def _forward(self, batch, **kwargs) -> CausalLMOutput:
        waveform = batch['waveform']
        transcription = batch['transcription']

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
        output = self.model(input_values, labels=labels, **kwargs)

        return output

    def _process_logits(self, logits: torch.Tensor):
        logits = F.softmax(logits, dim=-1)
        
        # If we consider speaker ids, sum over all spealer ids, and store result in node 32
        if logits.shape[-1] > 33:
            logits = torch.cat((logits[:, :, :32],
                                          torch.sum(logits[:, :, 33:], dim=-1).reshape((1, -1, 1))), dim=-1)
            
        predicted_ids = torch.argmax(logits, dim=-1)
        hypothesis = self.processor.batch_decode(predicted_ids)
        hypothesis = self.postprocessor(hypothesis)
        return hypothesis, logits

    def training_step(self, batch, batch_idx):
        output = self._forward(batch)
        self.log("train_loss", output.loss, batch_size=self.batch_size)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._forward(batch)
        hypothesis, _ = self._process_logits(output.logits)

        self.val_stats(hypothesis, batch['transcription'])
        self.log("val_loss", output.loss, batch_size=self.batch_size)
        return output.loss

    def validation_epoch_end(self, outputs: List[torch.Tensor]) -> None:
        self.log_dict(self.val_stats.compute_and_reset())
        return
    
    def test_step(self, batch: torch.Tensor, batch_idx):
        output = self._forward(batch, output_hidden_states=True)

        # Speech Recognition part
        transcription = batch["transcription"]
        hypothesis, processed_logits = self._process_logits(output.logits)
        for trans, hyp in zip(transcription, hypothesis):
            self.test_preds.append({
                "reference": trans,
                "hypothesis": hyp
            })

        # Speaker Identification part
        if self.save_embeddings:
            # Set all logit values where no speaker change is predicted to 0
            speaker_change_logits = torch.where(torch.argmax(processed_logits, dim=-1) == 32,
                                                processed_logits[-1][:,32], 0)

            # Apply max-pooling to find speaker change predictions with largest logit values
            _, max_speaker_change_idx = F.max_pool1d(speaker_change_logits, kernel_size=self.kernel_size, 
                                                     stride=self.kernel_size, padding=self.kernel_size // 2, 
                                                     return_indices=True)
            
            # Gather the speaker change logits of the found max indexes
            max_speaker_change_logits = torch.gather(speaker_change_logits, dim=1, index=max_speaker_change_idx)

            # Only use the indexes where the logit value has not been set to 0
            speaker_change_idx = max_speaker_change_idx[torch.where(max_speaker_change_logits != 0)]
            
            # Save the found speaker embeddings
            all_embeddings = output.hidden_states[-1].squeeze()
            self.embeddings = torch.cat(self.embeddings, all_embeddings[speaker_change_idx].to('cpu'))

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
    
    def reset_saves(self):
        self.embeddings: torch.Tensor = torch.Tensor()
        self.test_preds: List[Dict] = []
        return
