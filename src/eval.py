import click
import utils
import data.data as data
from config import Config

import torchaudio
from pathlib import Path
import models.wav2vec2_spch
import pytorch_lightning as pl
import torchaudio
import json


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
        Config.datapath, url="test-clean", download=True),
}


@click.command()
@click.option("--checkpoint_path", default=None)
def eval(checkpoint_path: str = None):
    device, _ = utils.set_device()
    
    # Load wav2vec2 module from checkpoint
    if checkpoint_path is not None:
        wav2vec2_module = models.wav2vec2_spch.Wav2Vec2Module.load_from_checkpoint(
            checkpoint_path)
        ckpt_version = int(Path(checkpoint_path).parts[-3][-7:])
    else:
        wav2vec2_module = models.wav2vec2_spch.Wav2Vec2Module()
        ckpt_version = 0

    print(f"Evaluating version {ckpt_version:0>7}:")

    wav2vec2_module.eval()
    wav2vec2_module = wav2vec2_module.to(device)
    all_res = []

    for dataset_str, dataset in datasets.items():
        if "test" in dataset_str:  # <- Remove this if we want to evaluate on test set
            continue

        dev_loader = data.initialize_loader(dataset, shuffle=False)

        print(f"Testing performance on dataset: {dataset_str}")
        trainer = pl.Trainer(log_every_n_steps=200, accelerator=device)
        res = trainer.test(model=wav2vec2_module, dataloaders=dev_loader)[0]

        res["dataset"] = dataset_str
        all_res.append(res)
        utils.json_dump(path=Path("logs") / "measures" /
                        f"{ckpt_version:0>7}-res.json", data=all_res)
        utils.write_dict_list(path=Path(
            "logs") / "preds" / f"{ckpt_version:0>7}-{dataset_str}-preds.csv", data=wav2vec2_module.test_preds)


if __name__ == "__main__":
    eval()
