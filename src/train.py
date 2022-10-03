
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from torch.utils.data.dataloader import DataLoader
from config import Config
import data
from tqdm import tqdm
import math
import jiwer
from collections import Counter
import utils
from pathlib import Path
import models.wav2vec2
import worderrorrate


class Trainer():
    def __init__(self, device: str, jobid: str) -> None:
        self.device = device
        self.jobid = jobid

        self.wup_epochs = 1
        self.wup_lr = 1e-3
        self.epochs = 10
        self.lr = 1e-5

        # Lists of results
        self.train_measures = []
        self.test_measures = []
        self.val_measures = []

        # Directory to which checkpoints are saved
        self.checkpoint_dir = Path(f"./checkpoints/{jobid}")

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader,
              model: Wav2Vec2ForCTC,
              processor: Wav2Vec2Processor):

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Warming-up
        self.warm_up(train_loader, model, processor)

        # Initialize optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.epochs)

        best_loss = 1e12
        for e in range(self.epochs):
            # Train model for one epoch
            train_measures = self._epoch(train_loader, model, processor, optim)
            self.train_measures.append(train_measures)
            self._write_results(type="train", measures=self.train_measures)
            scheduler.step()

            # Evaluate model performance
            self.eval(val_loader, model, processor)
            self._write_results(type="val", measures=self.val_measures)

            # Save model if loss is improved
            if self.val_measures[-1]["CTCloss"] < best_loss:
                torch.save(model.state_dict(),
                           self.checkpoint_dir / f"{self.jobid}.best.pt")

            # Save checkpoint every n steps
            if e % 10 == 0:
                torch.save(model.state_dict(), self.checkpoint_dir /
                           f"{self.jobid}.epoch_{e}.pt")

        return self.train_measures

    def eval(self, dataloader: DataLoader,
             model: Wav2Vec2ForCTC,
             processor: Wav2Vec2Processor):
        model.eval()
        with torch.no_grad():
            measures = self._epoch(dataloader, model, processor)
        model.train()
        self.val_measures.append(measures)
        return self.val_measures

    def warm_up(self, dataloader: DataLoader,
                model: Wav2Vec2ForCTC,
                processor: Wav2Vec2Processor):

        optim = torch.optim.Adam(model.parameters(), lr=self.wup_lr)

        # Freeze all layers except last
        model.requires_grad_(False)
        model.lm_head.requires_grad_(True)

        # Train model
        for _ in range(self.wup_epochs):
            measures = self._epoch(dataloader, model, processor, optim)
            self.train_measures.append(measures)
            self._write_results(type="train", measures=self.train_measures)

        # Unfreeze all layers and return
        model.requires_grad_(True)
        return self.train_measures

    # def evaluate(self, dataloader, model, processor, num_samples=20):
    #     res = []
    #     i = 0

    #     for sample in dataloader:
    #         waveform = sample['waveform'].to(self.device)
    #         ground_truth = sample['transcription']

    #         # Retrieve target labels
    #         with processor.as_target_processor():
    #             labels = processor(ground_truth, padding=True,
    #                                return_tensors='pt').input_ids
    #             labels = labels.to(self.device)

    #          # Retrieve input values
    #         input_values = processor(
    #             waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
    #         input_values = torch.reshape(
    #             input_values, (len(waveform), -1)).to(self.device)

    #         output = model(input_values, labels=labels)

    #         logits = output.logits
    #         predicted_ids = torch.argmax(logits, dim=-1)
    #         hypothesis = processor.batch_decode(predicted_ids)
    #         res.append({"ground_truth": ground_truth,
    #                    "hypothesis": hypothesis})

    #         i += 1
    #         if i == num_samples:
    #             break

    #     return res

    def _epoch(self, dataloader: DataLoader,
               model: Wav2Vec2ForCTC,
               processor: Wav2Vec2Processor,
               optim=None):

        total_loss = 0
        num_batches = 0
        measures = Counter({})
        # measures_no_sep = Counter({})

        pbar = tqdm(dataloader, desc=f"loss: ")
        for sample in pbar:
            waveform = sample['waveform'].to(self.device)
            reference = sample['transcription']
            # reference_no_sep = utils.remove_speaker_change_symbol(
            #     reference)

            # Retrieve input values
            input_values = processor(
                waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
            input_values = torch.reshape(
                input_values, (len(waveform), -1)).to(self.device)

            # Retrieve target labels
            with processor.as_target_processor():
                labels = processor(reference, padding=True,
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
            # hypothesis_no_sep = utils.remove_speaker_change_symbol(hypothesis)

            # Add measures
            # measures = Counter(jiwer.compute_measures(
            #     ground_truth, hypothesis)) + measures
            # measures_no_sep = Counter(jiwer.compute_measures(
            #     ground_truth_no_sep, hypothesis_no_sep)) + measures_no_sep
            measures = Counter(utils.spch_measure(
                reference[0], hypothesis[0])) + measures
            num_batches += 1

        # Construct measures dict and return
        # measures_no_sep = {f"no_sep_{k}": v for (
        #     k, v) in measures_no_sep.items()}
        # measures_no_sep = utils.mean_measures(
        #     measures_no_sep, num_batches=num_batches, len_dataset=len(dataloader.dataset))
        measures = utils.mean_measures(
            measures, num_batches, len_dataset=len(dataloader.dataset))
        # measures.update(measures_no_sep)
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
    ref = "# this is sentence and this is a longer sentence"
    hyp = "# nice"
    wer = worderrorrate.WER(ref, hyp)
    breakpoint()

    # Load model and processor
    processor = models.wav2vec2.load_processor()
    model = models.wav2vec2.load_model().to(device)

    # Load datasets
    train_set = data.CustomLibriSpeechDataset([
        Config.datapath + '/train-clean-no-rep',
        Config.datapath + '/train-clean-rep'])
    val_set = data.CustomLibriSpeechDataset([
        Config.datapath + '/val-clean-no-rep',
        Config.datapath + '/val-clean-rep'])
    test_set = data.CustomLibriSpeechDataset([
        Config.datapath + '/test-clean-no-rep',
        Config.datapath + '/test-clean-rep'])
    dev_set = data.CustomLibriSpeechDataset([
        Config.datapath + '/dev-clean-no-rep',
        Config.datapath + '/dev-clean-rep'])

    # Initialize dataloaders
    train_loader = data.initialize_loader(train_set)
    test_loader = data.initialize_loader(test_set)
    val_loader = data.initialize_loader(val_set)
    dev_loader = data.initialize_loader(dev_set)

    # Perform training
    trainer = Trainer(device=device, jobid=jobid)
    trainer.train(train_loader=train_loader, val_loader=val_loader,
                  model=model, processor=processor)
    trainer.eval(dev_loader, model, processor)


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
