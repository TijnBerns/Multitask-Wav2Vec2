import click
import utils
import data.custom_datasets as data
from config import Config

from pathlib import Path
import models.wav2vec2

import pytorch_lightning as pl
from typing import Optional, List
from evaluation.evaluator import SpeakerTrial, SpeakerRecognitionEvaluator, EmbeddingSample
from data.custom_datasets import LirbriSpeechItem, pad_collate
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import pickle
import torch


def get_datasets(trans_file: str):
    datasets = [
        # Development sets
        {
            "dev-no-rep": data.build_datapipe(Config.datapath + f'/dev-clean-no-rep/{trans_file}'),
            "dev-repA": data.build_datapipe(Config.datapath + f'/dev-clean-rep/{trans_file}'),
            "dev-repB": data.build_datapipe(Config.datapath + f'/dev-clean-repB/{trans_file}'),
            "dev-clean": data.build_datapipe(Config.datapath + f'/LibriSpeech/dev-clean.{trans_file}')
        },
        # Test sets
        {
            "test-no-rep": data.build_datapipe(Config.datapath + f'/test-clean-no-rep/{trans_file}'),
            "test-repA": data.build_datapipe(Config.datapath + f'/test-clean-rep/{trans_file}'),
            "test-repB": data.build_datapipe(Config.datapath + f'/test-clean-repB/{trans_file}'),
            "test-clean": data.build_datapipe(Config.datapath + f'/LibriSpeech/test-clean.{trans_file}')
        }
    ]
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
    device, _ = utils.set_device()
    wav2vec2_module.to(device)
    return wav2vec2_module, ckpt_version, checkpoint_path, prefix


def eval_spid(embedding_files: List[str], trials_path):
    trials = SpeakerTrial.from_file(trials_path)
    eer_dict = defaultdict(list)
    
    for hidden_state in range(12,13):
        len_dict = {}

        keys = []
        for embedding_file in embedding_files:
            embeddings = pickle.load(open(embedding_file, "rb"))[hidden_state]
            len_dict[embedding_file] = len(embeddings)
            keys.append(set([embedding.sample_id for embedding in embeddings]))
        
        # Compute intersection of all keys
        try:
            keys_intersection = set.intersection(*keys)
        except TypeError:
            return eer_dict

        for embedding_file in embedding_files:
            embeddings: List[EmbeddingSample] = pickle.load(
                open(embedding_file, "rb"))[hidden_state]
            embeddings_filtered = list(filter(
                lambda e: e.sample_id in keys_intersection, embeddings))

            print(f'Computing EER for {embedding_file} ({len(embeddings_filtered)}/{len(embeddings)} embeddings)...')
            eer = SpeakerRecognitionEvaluator.evaluate(
                pairs=trials,
                samples=embeddings_filtered,
                skip_eer=False,
                length_normalize=True,)
            print(f'EER:       {100 * eer:.2f}%\n' +
                f'Emb. used: {len(embeddings_filtered)}/{len(embeddings)}\n')
            eer_dict[str(embedding_file)].append(100 * eer)

    return eer_dict


def eval_asr(
    trans_file: str,
    vocab_path: str,
    checkpoint_path: str
):
    device, _ = utils.set_device()

    # Load datasets to evaluate on
    datasets = get_datasets(trans_file)
    datasets = {k: v for datasets in datasets for k, v in datasets.items()}

    # Load the module (from checkpoint)
    wav2vec2_module, ckpt_version, checkpoint_path, prefix = load_module(
        checkpoint_path, vocab_path)
    wav2vec2_module.eval()

    all_res = []
    save_string = f"{ckpt_version:0>7}-{prefix}.{trans_file[:-4]}"
    for dataset_str, dataset in datasets.items():
        wav2vec2_module.initMetrics(dataset_str.split('-')[0])

        # Initialize dataloader
        loader = data.initialize_loader(dataset, shuffle=False)

        # Perform test loop
        print(f"Testing performance on dataset: {dataset_str}")
        trainer = pl.Trainer(log_every_n_steps=200, accelerator=device)
        res = trainer.test(model=wav2vec2_module, dataloaders=loader)[0]

        # Evaluate speaker identification
        print(f"embeddings: {len(wav2vec2_module.embeddings)}")
        if len(wav2vec2_module.embeddings[0]) > 0:
            pickle.dump(wav2vec2_module.embeddings, open(
                Path("logs") / "embeddings" / f"{save_string}.{dataset_str}.embeddings.p", "wb"))

        # Add additional information to results dict
        res["dataset"] = dataset_str
        res["vocab"] = Path(vocab_path).name
        res["transcription"] = trans_file
        all_res.append(res)

        # Write results to out files
        utils.json_dump(path=Path("logs") / "measures" / f"{save_string}.res.json",
                        data=all_res)
        utils.write_dict_list(path=Path("logs") / "preds" / f"{save_string}.{dataset_str}.preds.csv",
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
        f"lightning_logs/version_{version_number}").rglob("*best*.ckpt")

    for checkpoint_path in checkpoints:
        eval_asr(trans_file, vocab_path, checkpoint_path)

    # Get paths to best embeddings
    dev_embedding_files = list(
        Path(f"logs/embeddings").rglob(f"{version_number}-best*dev*"))
    test_embedding_files = list(
        Path(f"logs/embeddings").rglob(f"{version_number}-best*test*"))

    # Evaluate SPID for given version
    eer_dev = eval_spid(embedding_files=dev_embedding_files,
                        trials_path=Path('/home/tberns/Speaker_Change_Recognition/trials/dev-clean.trials.txt'))
    eer_test = eval_spid(embedding_files=test_embedding_files,
                         trials_path=Path('/home/tberns/Speaker_Change_Recognition/trials/test-clean.trials.txt'))

    save_string = f"{version_number:0>7}-{'best'}.{trans_file[:-4]}"
    utils.json_dump(path=Path("logs") / "measures" /
                    f"{save_string}.eer.json", data={"dev": eer_dev, "test": eer_test})
    
    
# def temp():
#     long_sample = LirbriSpeechItem(
#         file_name="/ceph/csedu-scratch/other/tberns/afjiv.wav",
#         transcription=".",
#         speaker_id='t',
#         book_id='t',
#         utterance_id='t'
#     )
#     batch = pad_collate(long_sample)
#     checkpoint_path = "/home/tberns/Speaker_Change_Recognition/lightning_logs/version_2537373/checkpoints/epoch_0027.step_000074500.val-wer_0.0499.best.ckpt"
#     # checkpoint_path = "/home/tberns/Speaker_Change_Recognition/lightning_logs/version_2536477/checkpoints/epoch_0033.step_000095000.val-wer_0.0506.best.ckpt"
#     vocab_path = "/home/tberns/Speaker_Change_Recognition/src/models/vocab_spid.json"
#     wav2vec2_module, ckpt_version, checkpoint_path, prefix = load_module(
#         checkpoint_path, vocab_path)
#     output = wav2vec2_module.forward(batch)

#     logits = wav2vec2_module._preprocess_logits(output.logits)
#     hypothesis = wav2vec2_module._get_hypothesis(logits)
#     print(hypothesis)


if __name__ == "__main__":
    pl.seed_everything(Config.seed)
    eval_all()
    # temp()