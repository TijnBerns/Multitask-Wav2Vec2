import click
import utils
import data
import models

from config import Config
from train import Trainer

import pandas as pd
import torch
import torchaudio
from pathlib import Path

def eval(model_path: str, device: str, jobid: str, ) -> None:
    trainer = Trainer(device, jobid)
    processor = models.wav2vec2.load_processor()
    model = models.wav2vec2.load_model().to(device)
    model.load_state_dict(torch.load(model_path))
    model_jobid = Path(model_path).name.split('.')[0]

    dev_rep = data.CustomLibriSpeechDataset(
        [Config.datapath + '/dev-clean-rep'])
    dev_no_rep = data.CustomLibriSpeechDataset(
        [Config.datapath + '/dev-clean-no-rep'])
    dev_full = data.CustomLibriSpeechDataset(
        [Config.datapath + '/dev-clean-no-rep', Config.datapath + '/dev-clean-rep'])
    dev_clean = torchaudio.datasets.LIBRISPEECH(
        Config.datapath, url="dev-clean", download=True)

    dev_sets = {"dev_rep": dev_rep, "dev_no_rep": dev_no_rep,
                "dev_full": dev_full, "dev_clean": dev_clean}

    all_measures = []
    for (desc, dataset) in dev_sets.items():

        dataloader = data.initialize_loader(dataset)
        measures = trainer.eval(dataloader, model, processor)[-1]
        measures["dataset"] = desc
        all_measures.append(measures)

    df = pd.DataFrame.from_records(all_measures)
    df.to_csv(f"./logs/{model_jobid}.dev.results.csv")


@click.command()
@click.option("--model_path", help="Path to model checkpoint")
def main(model_path: str) -> None:
    device, jobid = utils.set_device()
    eval(device=device, jobid=jobid, model_path=model_path)


if __name__ == "__main__":
    main()
