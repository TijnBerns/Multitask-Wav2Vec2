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





@click.command()
@click.option("--checkpoint_path", default=None)
@click.option("--dataset", type=str, default=None,
              help="List or string of path(s) on which is evaluated.")
@click.option("--trans_path", default=None,
              help="List or string of path(s) in which train transcriptions are stored.")
@click.option("--vocab_path", default="models/vocab_spch.json", help="Path to the model vocab file.")
@click.option("--asr_only", type=bool, default=False)
def eval(dataset: str, trans_path: str, vocab_path: str, checkpoint_path: str, asr_only: bool=False):
    device, _ = utils.set_device()
    
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

    # Load wav2vec2 module from checkpoint
    if checkpoint_path is not None:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module.load_from_checkpoint(
            checkpoint_path, vocab_path=vocab_path)
        ckpt_version = int(Path(checkpoint_path).parts[-3][-7:])
    else:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module()
        ckpt_version = 0
        
    if asr_only:
        wav2vec2_module.trim_lm_head()
        
    # If dataset is provided only evaluate on the provided dataset
    if dataset is not None:
        if isinstance(dataset, str):
            dataset = [dataset]
        datasets = {"-".join(dataset): data.CustomLibriSpeechDataset(dataset, trans_path)}
    
    # Set transpath to "trans.csv" for logging purposes
    if trans_path is None:
        trans_path = "trans.csv"
      
    print(f"Evaluating version {ckpt_version:0>7}:")
    wav2vec2_module.eval()
    wav2vec2_module = wav2vec2_module.to(device)
    all_res = []

    for dataset_str, dataset in datasets.items():
        dev_loader = data.initialize_loader(dataset, shuffle=False)

        print(f"Testing performance on dataset: {dataset_str}")
        trainer = pl.Trainer(log_every_n_steps=200, accelerator=device)
        
        res = trainer.test(model=wav2vec2_module, dataloaders=dev_loader)[0]
        res["dataset"] = dataset_str
        res["vocab"] = Path(vocab_path).name[:-5]
        res["transcription"] = Path(trans_path).name[:-4]
        all_res.append(res)
        
        utils.json_dump(path=Path("logs") / "measures" /
                        f"{ckpt_version:0>7}-{Path(dataset_str).name}-{Path(trans_path).name[:-4]}-res.json", data=all_res)
        utils.write_dict_list(path=Path(
            "logs") / "preds" / f"{ckpt_version:0>7}-{Path(dataset_str).name}-{Path(trans_path).name[:-4]}-preds.csv", data=wav2vec2_module.test_preds)


if __name__ == "__main__":
    eval()
