import click
from torch.nn import ModuleList
import utils
import data.datasets as data
from config import Config

import torchaudio
from pathlib import Path
import models.wav2vec2
import pytorch_lightning as pl
import torchaudio
from typing import List, Union


def eval(trans_file: str, vocab_path: str, checkpoint_path: str):
    device, _ = utils.set_device()

    datasets = {
        "dev-full": data.CustomLibriSpeechDataset([
            Config.datapath + f'/dev-clean-no-rep/{trans_file}',
            Config.datapath + f'/dev-clean-rep/{trans_file}']),
        "dev-no-rep": data.CustomLibriSpeechDataset([
            Config.datapath + f'/dev-clean-no-rep/{trans_file}']),
        "dev-rep": data.CustomLibriSpeechDataset([
            Config.datapath + f'/dev-clean-rep/{trans_file}']),
        "dev-clean": torchaudio.datasets.LIBRISPEECH(
            Config.datapath, url="dev-clean", download=False),
        "test-full": data.CustomLibriSpeechDataset([
            Config.datapath + f'/test-clean-no-rep/{trans_file}',
            Config.datapath + f'/test-clean-rep/{trans_file}']),
        "test-no-rep": data.CustomLibriSpeechDataset([
            Config.datapath + f'/test-clean-no-rep/{trans_file}']),
        "test-rep": data.CustomLibriSpeechDataset([
            Config.datapath + f'/test-clean-rep/{trans_file}']),
        "test-clean": data.CustomLibriSpeechDataset([
            Config.datapath + f'LibriSpeech/test-clean-rep/{trans_file}'])
    }

    # Load wav2vec2 module from checkpoint
    if checkpoint_path is not None:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module.load_from_checkpoint(
            checkpoint_path, vocab_path=vocab_path)
        ckpt_version = int(Path(checkpoint_path).parts[-3][-7:])
    else:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module()
        ckpt_version = 0
        checkpoint_path = 'baseline    '

    wav2vec2_module.eval()
    wav2vec2_module = wav2vec2_module.to(device)
    all_res = []

    for dataset_str, dataset in datasets.items():
        if "test" in dataset_str:  # <- Remove this, if we want to evaluate on the test set
            continue

        dev_loader = data.initialize_loader(dataset, shuffle=False)

        print(f"Testing performance on dataset: {dataset_str}")
        trainer = pl.Trainer(log_every_n_steps=200, accelerator=device)

        res = trainer.test(model=wav2vec2_module, dataloaders=dev_loader)[0]
        res["dataset"] = dataset_str
        res["vocab"] = Path(vocab_path).name[:-5]
        res["transcription"] = trans_file[:-4]
        all_res.append(res)

        utils.json_dump(path=Path("logs") / "measures" /
                        f"{ckpt_version:0>7}-{trans_file[:-4]}-res.json", data=all_res)
        utils.write_dict_list(path=Path(
            "logs") / "preds" / f"{Path(checkpoint_path).name[:-4]}-{dataset_str}-{trans_file[:-4]}-preds.csv",
            data=wav2vec2_module.test_preds)


@click.command()
@click.option("--version_number", default=None)
@click.option("--trans_file", default='trans.csv',
              help="Name of transcription file on which model is evaluated.")
@click.option("--vocab_path", default="models/vocab_spch.json", help="Path to the model vocab file.")
def eval_all(version_number: str, trans_file: str, vocab_path: str):
    if version_number is not None:
        checkpoints = Path(
            f"../lightning_logs/version_{version_number}").rglob("*.ckpt")

        for checkpoint_path in checkpoints:
            print(f"Evaluating {checkpoint_path}")
            eval(trans_file, vocab_path, checkpoint_path)

    else:
        print(f"Evaluating base model")
        eval(trans_file, vocab_path, None)


if __name__ == "__main__":
    eval_all()
