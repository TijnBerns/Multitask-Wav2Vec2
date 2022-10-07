import click
import utils
import data
from config import Config

import pandas as pd
import torch
import torchaudio
from pathlib import Path
import models.wav2vec2
import pytorch_lightning as pl
import torchaudio
import re
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


datasets = {
    "dev-full": data.CustomLibriSpeechDataset([
        Config.datapath + '/dev-clean-no-rep',
        Config.datapath + '/dev-clean-rep']),
    "dev-no-rep": data.CustomLibriSpeechDataset([
        Config.datapath + '/dev-clean-no-rep']),
    "dev-rep": data.CustomLibriSpeechDataset([
        Config.datapath + '/dev-clean-rep']),
    "dev-clean": torchaudio.datasets.LIBRISPEECH(
        Config.datapath, url="dev-clean", download=True),
    "test-full": data.CustomLibriSpeechDataset([
        Config.datapath + '/test-clean-no-rep',
        Config.datapath + '/test-clean-rep']),
    "test-no-rep": data.CustomLibriSpeechDataset([
        Config.datapath + '/test-clean-no-rep']),
    "test-rep": data.CustomLibriSpeechDataset([
        Config.datapath + '/test-clean-rep']),
    "test-clean": torchaudio.datasets.LIBRISPEECH(
        Config.datapath, url="test-clean", download=True)
}


@click.command()
@click.option("--checkpoint_path", default=None, help="Path to model checkpoint. If None, finetuned wav2vec2 model is used.")
@click.option("--dataset_str", default="dev-full", help="the dataset on which the model is evaluated")
def eval(checkpoint_path: str, dataset_str: str):
    device, jobid = utils.set_device()
    
    dataset = datasets[dataset_str]
    dev_loader = data.initialize_loader(dataset)

    # Load wav2vec2 module from checkpoint
    if checkpoint_path is not None:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module.load_from_checkpoint(checkpoint_path)
        ckpt_version = int(Path(checkpoint_path).parts[-3][-7:])
    else:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module()

        
    wav2vec2_module.eval()
    wav2vec2_module = wav2vec2_module.to(device)
    trainer = pl.Trainer(log_every_n_steps=200, accelerator=device)
    trainer.test(model=wav2vec2_module, dataloaders=dev_loader)
    
    utils.write_dict_list(path=Path("logs") / "preds" / f"{ckpt_version:0>7}-{dataset_str}-preds.csv", data=wav2vec2_module.test_preds)
 

if __name__ == "__main__":
    eval()


# def eval(model_path: str, device: str, jobid: str, ) -> None:
#     trainer = Trainer(device, jobid)
#     processor = models.wav2vec2.load_processor()
#     model = models.wav2vec2.load_model().to(device)
#     model.load_state_dict(torch.load(model_path))
#     model_jobid = Path(model_path).name.split('.')[0]

#     dev_rep = data.CustomLibriSpeechDataset(
#         [Config.datapath + '/dev-clean-rep'])
#     dev_no_rep = data.CustomLibriSpeechDataset(
#         [Config.datapath + '/dev-clean-no-rep'])
#     dev_full = data.CustomLibriSpeechDataset(
#         [Config.datapath + '/dev-clean-no-rep', Config.datapath + '/dev-clean-rep'])
#     dev_clean = torchaudio.datasets.LIBRISPEECH(
#         Config.datapath, url="dev-clean", download=True)

#     dev_sets = {"dev_rep": dev_rep, "dev_no_rep": dev_no_rep,
#                 "dev_full": dev_full, "dev_clean": dev_clean}

#     all_measures = []
#     for (desc, dataset) in dev_sets.items():

#         dataloader = data.initialize_loader(dataset)
#         measures = trainer.eval(dataloader, model, processor)[-1]
#         measures["dataset"] = desc
#         all_measures.append(measures)

#     df = pd.DataFrame.from_records(all_measures)
#     df.to_csv(f"./logs/{model_jobid}.dev.results.csv")


# @click.command()
# @click.option("--model_path", help="Path to model checkpoint")
# def main(model_path: str) -> None:
#     device, jobid = utils.set_device()
#     eval(device=device, jobid=jobid, model_path=model_path)
