import click
import utils
import data
import models

from config import Config
from train import Trainer

import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor



# @click.command()
# @click.option("--model_path", help="Path to model checkpoint")
def main(device: str, jobid: str, model_path: str = ""):
    trainer = Trainer(device, jobid)
    processor = models.wav2vec2.load_processor()
    model = models.wav2vec2.load_model().to(device)
    model.load_state_dict(torch.load(model_path))
    # processor = Wav2Vec2Processor.from_pretrained(
    #     "facebook/wav2vec2-base-960h")
    # model = Wav2Vec2ForCTC.from_pretrained(
    #     "facebook/wav2vec2-base-960h", ctc_loss_reduction="mean").to(device)

    test_sets = {"test-clean": torchaudio.datasets.LIBRISPEECH(Config.datapath, url="test-clean", download=True),
                 "merged-test-clean-no-rep": data.CustomLibriSpeechDataset(Config.datapath + '/test-clean-no-rep')}

    all_measures = []
    for (desc, dataset) in test_sets.items():

        dataloader = data.initialize_loader(dataset)
        measures = trainer.test(dataloader, model, processor)
        measures["dataset"] = desc
        all_measures.append(measures)

    df = pd.DataFrame.from_records(all_measures)
    df.to_csv(f"./logs/{jobid}_test_results.csv")


if __name__ == "__main__":
    device, jobid = utils.set_device()
    main(device, jobid)
