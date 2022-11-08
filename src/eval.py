import click
import utils
import data.datasets as data
from config import Config

from pathlib import Path
import models.wav2vec2

import pytorch_lightning as pl
from typing import Optional
from evaluation.evaluator import SpeakerTrial, SpeakerRecognitionEvaluator
from tqdm import tqdm
from pprint import pprint


def get_datasets(trans_file: str):
    datasets = {
        # "dev-mini": data.build_datapipe(Config.datapath + f'/dev-clean-no-rep/dev-mini.csv'),
        # "train-mini": data.build_datapipe(Config.datapath + f'/train-clean-100-no-rep/mini.csv'),
        # "dev-full": data.build_datapipe((
        #     Config.datapath + f'/dev-clean-no-rep/{trans_file}',
        #     Config.datapath + f'/dev-clean-rep/{trans_file}'
        # )),
        "dev-no-rep": data.build_datapipe(Config.datapath + f'/dev-clean-no-rep/{trans_file}'),
        "dev-rep": data.build_datapipe(Config.datapath + f'/dev-clean-rep/{trans_file}'),
        "dev-clean": data.build_datapipe(Config.datapath + f'/LibriSpeech/dev-clean.{trans_file}'),
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
    return datasets


def load_module(checkpoint_path: Optional[str], vocab_path: Optional[str]):
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
    return wav2vec2_module, ckpt_version, checkpoint_path, prefix


def eval_spid(trials, wav2vec2_module):
    # For each trial, compute scores based on cosine similarity
    eer_list = []
    # for i, embedding_list in enumerate(wav2vec2_module.embeddings):
    # embedding_list = wav2vec2_module.embeddings[i]
    embedding_list = wav2vec2_module.embeddings
    i = 12

    print(f"Computing scores...")
    eer = SpeakerRecognitionEvaluator.evaluate(
        pairs=trials,
        samples=embedding_list,
        skip_eer=False,
        length_normalize=True,
        # mean_embedding=wav2vec2_module.mean_embedding,
        # std_embedding=wav2vec2_module.std_embedding
    )
    print(f'computed eer for layer {i}:\t {100 * eer}%')
    eer_list.append(eer)

    return eer_list


def eval_asr(
    trans_file: str,
    vocab_path: str,
    checkpoint_path: str
):
    device, _ = utils.set_device()

    # Load datasets to evaluate on
    datasets = get_datasets(trans_file)

    # Load the module (from checkpoint)
    wav2vec2_module, ckpt_version, checkpoint_path, prefix = load_module(
        checkpoint_path, vocab_path)
    wav2vec2_module.eval()
    wav2vec2_module = wav2vec2_module.to(device)

    all_res = []
    for dataset_str, dataset in datasets.items():
        # Initialize dataloader
        loader = data.initialize_loader(dataset, shuffle=False)

        # Perform test loop
        print(f"Testing performance on dataset: {dataset_str}")
        trainer = pl.Trainer(log_every_n_steps=200, accelerator=device)
        res = trainer.test(model=wav2vec2_module, dataloaders=loader)[0]

        # Evaluate speaker identification
        if len(wav2vec2_module.embeddings) > 0:
            trials = SpeakerTrial.from_file(
                Path('/home/tberns/Speaker_Change_Recognition/trials/dev-clean.trials.txt'))
            eer = eval_spid(trials, wav2vec2_module)
            res["EER"] = eer

        # Add additional information to results dict
        res["dataset"] = dataset_str
        res["vocab"] = Path(vocab_path).name
        res["transcription"] = trans_file
        all_res.append(res)

        # Write results to out files
        pprint(res)
        utils.json_dump(path=Path("logs") / "measures" /
                        f"{ckpt_version:0>7}-{trans_file[:-4]}-res.json", data=all_res)
        utils.write_dict_list(path=Path(
            "logs") / "preds" / f"{ckpt_version:0>7}-{prefix}-{dataset_str}-{trans_file[:-4]}-preds.csv",
            data=wav2vec2_module.test_preds)

        # Reset the saved embeddings and predictions
        wav2vec2_module.reset_saves()


@click.command()
@click.option("--version_number",
              default=None,
              help="Version number of which the model is evaluated.")
@click.option("--trans_file",
              default='trans-st.csv',
              help="Name of transcription file on which model is evaluated (not the complete path).")
@click.option("--vocab_file",
              default="vocab_spid.json",
              help="name of the model vocab file (not the complete path).")
def eval_all(
    version_number: str,
    trans_file: str,
    vocab_file: str
):
    vocab_path = f"src/models/{vocab_file}"
    if version_number is None:
        print(f"Evaluating base model")
        eval_asr(trans_file, vocab_path, None)
        return

    checkpoints = Path(
        f"lightning_logs/version_{version_number}").rglob("*.ckpt")

    for checkpoint_path in checkpoints:
        # if ".last" in str(checkpoint_path):
        #     continue

        print(f"Evaluating {checkpoint_path}")
        eval_asr(trans_file, vocab_path, checkpoint_path)


if __name__ == "__main__":
    eval_all()
