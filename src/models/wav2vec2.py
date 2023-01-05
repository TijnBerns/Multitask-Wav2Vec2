from config import Config
import utils
from typing import List, Any, Dict, Tuple, Union, Optional, Set
from evaluation.metrics import SpeakerChangeStats
from collections import defaultdict, deque

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
from models.tri_stage import TriStageLearningRateLambdaLRFunction as TriStageLR
from data.custom_datasets import LirbriSpeechBatch
from evaluation.evaluator import EmbeddingSample
import numpy as np


def load_processor(vocab_path: str) -> Tuple[Wav2Vec2Processor, int]:
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
                 stage: int = 2,
                 num_steps_stage_one: int = Config.num_steps_stage_one,
                 num_steps_stage_two: int = Config.num_steps_stage_two,
                 lr_stage_one: float = Config.lr_stage_one,
                 lr_stage_two: float = Config.lr_stage_two,
                 batch_size: int = Config.batch_size,
                 kernel_size: int = 10,
                 postprocessor: PostProcessor = StripSpeakerChange(),
                 dataset_type: str = 'dev'):

        super().__init__()
        self.processor, self.vocab_size = load_processor(vocab_path)
        self.model: Wav2Vec2ForCTC = load_model(self.vocab_size)
        self.stage: int = stage
        self.postprocessor = postprocessor

        # Speaker embeddings
        self.save_embeddings: bool = self.vocab_size > 32
        self.embeddings = defaultdict(list)
        self.kernel_size: int = kernel_size
        self.embeddings_queue = deque(maxlen=3000)
        # self.mean_embedding = torch.nn.parameter.Parameter(
        #     data=torch.zeros(768), requires_grad=False
        # )
        # self.std_embedding = torch.nn.parameter.Parameter(
        #     data=torch.ones(768), requires_grad=False
        # )
        self.mean_embedding = torch.zeros(768)
        self.std_embedding = torch.zeros(768)

        # Metrics
        self.initMetrics(dataset_type)

        # Training parameters
        self.num_steps_stage_one: int = num_steps_stage_one
        self.num_steps_stage_two: int = num_steps_stage_two
        self.lr_stage_one: float = lr_stage_one
        self.lr_stage_two: float = lr_stage_two
        self.batch_size: int = batch_size

        # temporary attributes
        
        self.size_mismatch_count = 0
        
    def initMetrics(self, dataset_type):
        self.val_stats: SpeakerChangeStats = SpeakerChangeStats(prefix="val")
        self.test_stats: SpeakerChangeStats = SpeakerChangeStats(prefix=dataset_type)
        self.test_preds: List[Dict] = []
        

    def forward(self, batch: LirbriSpeechBatch, **kwargs) -> CausalLMOutput:
        waveforms = batch.waveforms
        transcriptions = batch.transcriptions
        # attention_mask = torch.tensor(waveforms != 0, dtype=torch.long)

        # Retrieve input values
        input_values = self.processor(
            waveforms, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
        input_values = torch.reshape(
            input_values, (len(waveforms), -1)).to(self.device)

        # Retrieve target labels
        with self.processor.as_target_processor():
            labels = self.processor(transcriptions, padding=True,
                                    return_tensors='pt').input_ids
            labels = labels.to(self.device)

        # Compute loss by passing labels
        output = self.model(input_values, labels=labels, **kwargs)

        return output

    def training_step(self, batch: LirbriSpeechBatch, batch_idx):
        output = self.forward(batch)
        
        self.log("train_loss", output.loss, batch_size=self.batch_size)
        return output.loss

    def validation_step(self, batch: LirbriSpeechBatch, batch_idx):
        output = self.forward(batch)

        # Process the logits to obtain transcription
        logits = self._preprocess_logits(output.logits)
        hypothesis = self._get_hypothesis(logits)

        # Update and log validation stats
        self.val_stats(hypothesis, batch.transcriptions)
        self.log("val_loss", output.loss, batch_size=self.batch_size)
        return output.loss

    def validation_epoch_end(self, outputs: List[torch.Tensor]) -> None:
        self.log_dict(self.val_stats.compute_and_reset())
        return

    def test_step(self, batch: LirbriSpeechBatch, batch_idx):
        output = self.forward(batch, output_hidden_states=True)
        transcription = batch.transcriptions

        # Process the logits to obtain transcription
        processed_logits = self._preprocess_logits(output.logits)
        hypothesis = self._get_hypothesis(processed_logits)
        for trans, hyp in zip(transcription, hypothesis):
            self.test_preds.append({
                "reference": trans,
                "hypothesis": hyp
            })
            
        # Update and log statistics
        self.test_stats(hypothesis, transcription)
        self.log("test_loss", output.loss, batch_size=self.batch_size)

        if not self.save_embeddings:
            return output.loss

        # Save speaker embedddings
        for i in range(len(batch)):
            speaker_change_idx = self._extract_embeddings(
                output.hidden_states, processed_logits[i])

            reconstruced_keys, embbeding_idx = utils.reconstruct_keys(
                batch.keys[i])
            
            if len(set(embbeding_idx)) != len(speaker_change_idx):
                self.size_mismatch_count += 1
                return output.loss

            # Add found embeddings and their corresponding keys to lists
            for k, hidden_states in enumerate(output.hidden_states):
                for j in embbeding_idx:
                    embedding = hidden_states.squeeze()[speaker_change_idx][j].to('cpu')
                    key = reconstruced_keys[j]
                    self.embeddings[k].append(EmbeddingSample(key, embedding))
                
            self.embeddings_queue.extend(speaker_change_idx)

        return output.loss

    def test_epoch_end(self, outputs: Any) -> None:
        self.log_dict(self.test_stats.compute_and_reset())
        # update the mean & std of test embeddings
        # if self.save_embeddings:
        #     mean, std = self._compute_mean_std_batch(
        #         [e for e in self.embeddings_queue])

        #     with torch.no_grad():
        #         mean = mean.to(self.mean_embedding.device)
        #         std = std.to(self.std_embedding.device)

        #         self.mean_embedding[:] = mean
        #         self.std_embedding[:] = std
        print(self.size_mismatch_count)
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
                self.model.parameters(), lr=self.lr_stage_one)
            schedule = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.num_steps_stage_one,
                    eta_min=0),
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": None,
            }
        elif self.stage == 2:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr_stage_two)
            # setup the learning rate schedule.
            schedule = {
                # Required: the scheduler instance.edu
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    TriStageLR(max_steps=self.num_steps_stage_two,
                               warmup_stage_ratio=0.1,
                               constant_stage_ratio=0.4,
                               decay_stage_ratio=0.5,
                               initial_lr=self.lr_stage_two / 100,
                               base_lr=self.lr_stage_two,
                               final_lr=self.lr_stage_two / 20)),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after an optimizer update.    # def _extract_embeddings_batch(self, hidden_states: torch.Tensor, processed_logits: torch.Tensor):
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "lr",
            }
        else:
            raise NotImplementedError(
                f"The stage of the model must be either 1 or 2 but got {self.stage}")

        return [optimizer], [schedule]

    def reset_saves(self):
        self.embeddings = defaultdict(list)
        self.test_preds = defaultdict(list) 
        return

    def _compute_mean_std_batch(self, all_tensors: List[torch.Tensor]):
        # compute mean and std over each dimension of EMBEDDING_SIZE
        # with a tensor of shape [NUM_SAMPLES, EMBEDDING_SIZE]
        stacked_tensors = torch.stack(all_tensors)
        print(stacked_tensors.shape)

        std, mean = torch.std_mean(stacked_tensors, dim=0)

        return mean, std

    def _preprocess_logits(self, logits: torch.Tensor):
        logits = F.softmax(logits, dim=-1)

        # If we consider speaker ids, sum over all spealer ids, and store result in node 32
        if logits.shape[-1] > 33:
            logits = torch.cat((logits[:, :, :32], torch.sum(
                logits[:, :, 33:], dim=-1).reshape((*logits.shape[:2], 1))), dim=-1)
        return logits

    def _get_hypothesis(self, logits):
        predicted_ids = torch.argmax(logits, dim=-1)
        hypothesis = self.processor.batch_decode(predicted_ids)
        hypothesis = self.postprocessor(hypothesis)
        return hypothesis

    def _extract_embeddings(self, hidden_states: Tuple[torch.Tensor], processed_logits):
        # Set all logit values where no speaker change is predicted to 0
        speaker_change_logits = torch.where(torch.argmax(processed_logits, dim=-1) == 32, processed_logits[:, 32], 0)

        # Apply max-pooling to find speaker change predictions with largest logit values
        speaker_change_logits = speaker_change_logits.reshape(1, -1)
        _, max_speaker_change_idx = F.max_pool1d(speaker_change_logits,
                                                 kernel_size=self.kernel_size, stride=self.kernel_size,
                                                 padding=self.kernel_size // 2,
                                                 return_indices=True)

        # Gather the speaker change logits of the found max indexes
        max_speaker_change_logits = torch.gather(speaker_change_logits, dim=1, index=max_speaker_change_idx)

        # Only use the indexes where the logit value has not been set to 0
        speaker_change_idx: torch.Tensor = max_speaker_change_idx[torch.where(max_speaker_change_logits != 0)]

        if speaker_change_idx.shape[0] == 0:
            speaker_change_idx = torch.tensor([0])

        # Save the found speaker embeddings
        return speaker_change_idx

    def _add_batch_to_embedding_queue(self, embedding: torch.Tensor):
        # make sure to keep it into CPU memory
        embedding = embedding.detach().to("cpu")

        # unbind embedding of shape [BATCH_SIZE, EMBEDDING_SIZE] into a list of
        # tensors of shape [EMBEDDING_SIZE] with len=BATCH_SIZE
        embedding_list = torch.unbind(embedding, dim=0)

        self.embeddings_queue.extend(
            [embedding for embedding in embedding_list])

    # def _extract_embeddings_batch(self, hidden_states: torch.Tensor, processed_logits: torch.Tensor):
    #     # Set all logit values where no speaker change is predicted to 0
    #     speaker_change_logits = torch.where(torch.argmax(
    #         processed_logits, dim=-1) == 32, processed_logits[:, :, 32], 0)

    #     # Apply max-pooling to find speaker change predictions with largest logit values
    #     # speaker_change_logits = speaker_change_logits.reshape(1, -1)
    #     _, max_speaker_change_idx = F.max_pool1d(
    #         speaker_change_logits, kernel_size=self.kernel_size, stride=self.kernel_size, padding=self.kernel_size // 2, return_indices=True)

    #     # Gather the speaker change logits of the found max indexes
    #     max_speaker_change_logits = torch.gather(
    #         speaker_change_logits, dim=1, index=max_speaker_change_idx)

    #     # Only use first found speaker change, if not found use first index
    #     speaker_change_idx = torch.where(torch.max(
    #         max_speaker_change_logits) > 0, torch.argmax(max_speaker_change_logits, dim=-1), 0)

    #     return hidden_states[-1][torch.arange(hidden_states[-1].size(0)), speaker_change_idx].to('cpu')
