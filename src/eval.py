import click
from torch import embedding
from torch.nn import ModuleList
import utils
import data.datasets as data
from config import Config

import torchaudio
from pathlib import Path
import models.wav2vec2
from models.processor import StripSpeakerChange
import pytorch_lightning as pl
import torchaudio
from typing import List, Union



def eval_spid(vocab_path: str, checkpoint_path: str, shards_dirs: str, trial_lists: str):
    pairs = list()
    embeddings = list()
    
    for shard_dir, trial_list in zip(shards_dirs, trial_lists):
        pass
    
        
def eval_asr(trans_file: str, vocab_path: str, checkpoint_path: str):
    device, _ = utils.set_device()

    datasets = {
        "dev-full": data.CustomLibriSpeechDataset([
            Config.datapath + f'/dev-clean-no-rep/{trans_file}',
            Config.datapath + f'/dev-clean-rep/{trans_file}']),
        # "dev-no-rep": data.CustomLibriSpeechDataset([
        #     Config.datapath + f'/dev-clean-no-rep/{trans_file}']),
        # "dev-rep": data.CustomLibriSpeechDataset([
        #     Config.datapath + f'/dev-clean-rep/{trans_file}']),
        "dev-clean": data.CustomLibriSpeechDataset([
            Config.datapath + f'/LibriSpeech/dev-clean.{trans_file}']),
        # "test-full": data.CustomLibriSpeechDataset([
        #     Config.datapath + f'/test-clean-no-rep/{trans_file}',
        #     Config.datapath + f'/test-clean-rep/{trans_file}']),
        # "test-no-rep": data.CustomLibriSpeechDataset([
        #     Config.datapath + f'/test-clean-no-rep/{trans_file}']),
        # "test-rep": data.CustomLibriSpeechDataset([
        #     Config.datapath + f'/test-clean-rep/{trans_file}']),
        # "test-clean": data.CustomLibriSpeechDataset([
        #     Config.datapath + f'/LibriSpeech/test-clean.{trans_file}']),
        # "train-no-rep": data.CustomLibriSpeechDataset([
        #     Config.datapath + f'/train-clean-100-no-rep/{trans_file}']),
    }
    
    if checkpoint_path is not None:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module.load_from_checkpoint(
            checkpoint_path, vocab_path=vocab_path)
        ckpt_version = int(Path(checkpoint_path).parts[-3][-7:]) 
        prefix = Path(checkpoint_path).name.split('.')[-2] 
    else:
        wav2vec2_module = models.wav2vec2.Wav2Vec2Module()
        ckpt_version = 0
        checkpoint_path = 'baseline----'
        prefix = 'baseline'

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
        res["transcription"] = trans_file[:-4]
        all_res.append(res)

        utils.json_dump(path=Path("logs") / "measures" /
                        f"{ckpt_version:0>7}-{trans_file[:-4]}-res.json", data=all_res)
        utils.write_dict_list(path=Path(
            "logs") / "preds" / f"{ckpt_version:0>7}-{prefix}-{dataset_str}-{trans_file[:-4]}-preds.csv",
            data=wav2vec2_module.test_preds)
        wav2vec2_module.reset_saves()


@click.command()
@click.option("--version_number", default=None)
@click.option("--trans_file", default='trans.csv',
              help="Name of transcription file on which model is evaluated.")
@click.option("--vocab_file", default="vocab_spid.json", help="Path to the model vocab file.")
def eval_all(version_number: str, trans_file: str, vocab_file: str):
    vocab_path = f"src/models/{vocab_file}"
    if version_number is None:
        print(f"Evaluating base model")
        eval_asr(trans_file, vocab_path, None)
        return

    checkpoints = Path(
        f"lightning_logs/version_{version_number}").rglob("*.ckpt")

    for checkpoint_path in checkpoints:
        if ".best" in str(checkpoint_path):
            continue
        
        print(f"Evaluating {checkpoint_path}")
        eval_asr(trans_file, vocab_path, checkpoint_path)


if __name__ == "__main__":
    eval_all()
