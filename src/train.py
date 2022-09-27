
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from config import Config
import data
from tqdm import tqdm
import math
import jiwer
from collections import Counter
import utils
from pathlib import Path
import models.wav2vec2


class Trainer():
    def __init__(self, device, jobid) -> None:
        self.device = device
        self.jobid = jobid

        self.train_measures = []
        self.val_measures = []
        self.test_measures = []
        self.checkpoint_dir = Path(f"./checkpoints/{jobid}")

    def train(self, train_loader, val_loader, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, num_epochs: int):
        # Create directory in which checkpoints are saved
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, num_epochs)

        best_loss = 1e12
        for e in range(num_epochs):
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

            # Save model if loss is improved
            if val_measures["CTCloss"] < best_loss:
                torch.save(model.state_dict(),
                           self.checkpoint_dir / f"{self.jobid}.best.pt")

            # Save checkpoint every n steps
            if e % 10 == 0:
                torch.save(model.state_dict(), self.checkpoint_dir /
                           f"{self.jobid}.epoch_{e}.pt")

        return

    def test(self, test_loader, model, processor, type: str = "test"):
        measures = self._epoch(test_loader, model, processor)
        self.test_measures.append(measures)
        # self._write_results(type=type, measures=measures)
        return measures

    def evaluate(self, dataloader, model, processor, num_samples=20):
        res = []
        i = 0

        for sample in dataloader:
            waveform = sample['waveform'].to(self.device)
            ground_truth = sample['transcription']

            # Retrieve target labels
            with processor.as_target_processor():
                labels = processor(ground_truth, padding=True,
                                   return_tensors='pt').input_ids
                labels = labels.to(self.device)

             # Retrieve input values
            input_values = processor(
                waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
            input_values = torch.reshape(
                input_values, (len(waveform), -1)).to(self.device)

            output = model(input_values, labels=labels)

            logits = output.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            hypothesis = processor.batch_decode(predicted_ids)
            res.append({"ground_truth": ground_truth,
                       "hypothesis": hypothesis})

            i += 1
            if i == num_samples:
                break

        return res

    def _epoch(self, dataloader, model, processor, optim=None):
        total_loss = 0
        num_batches = 0
        measures = Counter({})
        measures_no_sep = Counter({})

        pbar = tqdm(dataloader, desc=f"loss: ")
        for sample in pbar:
            waveform = sample['waveform'].to(self.device)
            ground_truth = sample['transcription']
            ground_truth_no_sep = utils.remove_speaker_change_symbol(
                ground_truth)

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
            pbar.set_description(f"loss: {loss}")

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
            hypothesis_no_sep = utils.remove_speaker_change_symbol(hypothesis)

            # Add measures
            measures = Counter(jiwer.compute_measures(
                ground_truth, hypothesis)) + measures
            measures_no_sep = Counter(jiwer.compute_measures(
                ground_truth_no_sep, hypothesis_no_sep)) + measures_no_sep

            num_batches += 1

        # Construct measures dict and return
        measures_no_sep = {f"no_sep_{k}": v for (
            k, v) in measures_no_sep.items()}
        measures_no_sep = utils.mean_measures(
            measures_no_sep, num_batches=num_batches, len_dataset=len(dataloader.dataset))
        measures = utils.mean_measures(
            measures, num_batches, len_dataset=len(dataloader.dataset))
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

        return


def main(device: str, jobid: str):
    # Load model and processor
    processor = models.wav2vec2.load_processor()
    model = models.wav2vec2.load_model().to(device)

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
                  model=model, processor=processor, num_epochs=60)
    trainer.test(test_loader, model, processor)


if __name__ == "__main__":
    device, jobid = utils.set_device()
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
