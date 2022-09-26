
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import torch
from config import Config
import data
from tqdm import tqdm
import math
import jiwer
from collections import Counter
from utils import mean_measures, remove_speaker_change_symbol
from pathlib import Path
import os


class Trainer():
    def __init__(self, device, jobid) -> None:
        self.device = device
        self.jobid = jobid

        # self.lr = lr
        self.train_measures = []
        self.val_measures = []
        self.test_measures = []

    def train(self, train_loader, val_loader, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, num_epochs: int):
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, num_epochs)

        for _ in range(num_epochs):
            # Train model for one epoch
            train_measures = self._epoch(train_loader, model, processor, optim)
            scheduler.step()
            self.train_measures.append(train_measures)
            self._write_results(type="train", measures=self.train_measures)

            # Evaluate model performance
            model.eval()
            with torch.no_grad():
                val_measures = self._epoch(val_loader, model, processor)
                self.val_measures.append(val_measures)
                self._write_results(type="val", measures=self.val_measures)
        return

    def test(self, test_loader, model, processor, type: str = "test"):
        measures = self._epoch(test_loader, model, processor)
        measures.append(measures)
        self._write_results(type=type, measures=measures)
        return measures

    def _epoch(self, dataloader, model, processor, optim=None):
        total_loss = 0
        num_batches = 0
        measures = Counter({})
        measures_no_sep = Counter({})

        for sample in tqdm(dataloader):
            waveform = sample['waveform'].to(self.device)
            ground_truth = sample['transcription']
            ground_truth_no_sep = remove_speaker_change_symbol(ground_truth)

            # Retrieve input values
            input_values = processor(
                waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
            input_values = torch.reshape(
                input_values, (len(waveform), -1)).to(self.device)

            # Retrieve target labels
            with processor.as_target_processor():
                labels = processor(ground_truth, padding=True,
                                   return_tensors='pt').input_ids
                labels = labels.to(self.device)

            # Compute loss by passing labels
            output = model(input_values, labels=labels)
            loss = output.loss
            total_loss += loss.item()

            # Check for nan loss
            if math.isnan(loss):
                exit("LOSS REACHED NAN VALUE")

            # Backpropagation
            if optim is not None:
                optim.zero_grad()
                loss.backward()
                optim.step()

            # Construct hypothesis
            logits = output.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            hypothesis = processor.batch_decode(predicted_ids)
            hypothesis_no_sep = remove_speaker_change_symbol(hypothesis)

            # Add measures
            measures = Counter(jiwer.compute_measures(
                ground_truth, hypothesis)) + measures
            measures_no_sep = Counter(jiwer.compute_measures(
                ground_truth_no_sep, hypothesis_no_sep)) + measures_no_sep

            num_batches += 1

        # Construct measures dict and return
        measures_no_sep = {f"no_sep_{k}": v for (
            k, v) in measures_no_sep.items()}
        measures_no_sep = mean_measures(measures_no_sep, num_batches)
        measures = mean_measures(measures, num_batches)
        measures.update(measures_no_sep)
        measures['CTCloss'] = total_loss / len(dataloader.dataset)

        return measures

    def _write_results(self, type: str, measures: list):
        assert type in ["train", "val", "test", "dev"]
        log_file = Path(Config.logpath) / "measures" / \
            (str(self.jobid) + f".{type}.json")

        if not log_file.exists():
            log_file.touch(exist_ok=False)

        with open(log_file, "w") as f:
            json.dump(measures, f, indent=2)


def main(device: str, jobid: str):
    # Initialize processor
    tokenizer = Wav2Vec2CTCTokenizer(
        "src/models/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=Config.sample_rate, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load model
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h", ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id).to(device)
    model.freeze_feature_encoder()

    # Load datasets
    train_set = data.CustomLibriSpeechDataset(
        Config.datapath + '/train-clean-no-rep')
    val_set = data.CustomLibriSpeechDataset(
        Config.datapath + '/val-clean-no-rep')
    test_set = data.CustomLibriSpeechDataset(
        Config.datapath + '/test-clean-no-rep')
    dev_set = data.CustomLibriSpeechDataset(
        Config.datapath + '/dev-clean-no-rep')

    # Initialize dataloaders
    train_loader = data.initialize_loader(train_set)
    test_loader = data.initialize_loader(test_set)
    val_loader = data.initialize_loader(val_set)
    dev_loader = data.initialize_loader(dev_set)

    # Perform training
    trainer = Trainer(device=device, jobid=jobid)
    trainer.train(train_loader=train_loader, val_loader=val_loader,
                  model=model, processor=processor, num_epochs=10)
    trainer.test(test_loader, model, processor)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    # device = 'cpu'
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)

    if jobid == default_jobid and device != "cpu":
        exit("Running on GPU without use of slurm!")

    main(device, jobid)


"""
for sample in dev_clean_loader:

    input_values = processor(sample['waveform'], sampling_rate=16_000, return_tensors="pt", padding="longest").input_values  
    
    # retrieve logits
    input_values = torch.reshape(input_values, (1,-1))
    logits = model(input_values).logits
    

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    breakpoint()
        
"""
