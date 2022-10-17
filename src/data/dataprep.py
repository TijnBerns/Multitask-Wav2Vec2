import sys
sys.path.append('src')

from data.pairgenerator import PairGeneratorRepeat, PairGeneratorNoRepeat
import data.utils
import data.datasets as datasets
from typing import List, Union, Dict
from config import Config
import random
from pathlib import Path
import click



def merge_dataset(dataset, dataset_str: str, num_samples: int) -> None:
    root = Path(Config.datapath)
    pair_generator_repeat = PairGeneratorRepeat(num_samples=num_samples,
                                                min_tokens=Config.min_tokens,
                                                max_tokens=Config.max_tokens,
                                                max_attempts=Config.max_attempts)

    pair_generator_no_repeat = PairGeneratorNoRepeat(num_samples=num_samples,
                                                     min_tokens=Config.min_tokens,
                                                     max_tokens=Config.max_tokens,
                                                     max_attempts=Config.max_attempts)

    for generator, suffix in [(pair_generator_repeat, "-rep"),
                              (pair_generator_no_repeat, "-no-rep")]:
        print(f"Generating {dataset_str}{suffix}")
        merged_pairs = generator.gen_pairs(dataset)
        data.utils.save_pairs(root / (dataset_str + suffix), merged_pairs)


def transcribe_libri_clean():
    librispeach_clean_path = Path(Config.datapath) / "LibriSpeech"
    train_clean_100_path = librispeach_clean_path / 'train-clean-100'
    test_clean_path = librispeach_clean_path / 'test-clean'
    dev_clean_path = librispeach_clean_path / 'dev-clean'

    datapaths = [train_clean_100_path, test_clean_path, dev_clean_path]

    for path in datapaths:
        for trans_file in list(path.glob("*trans.csv")):
            trans_file.unlink()

        data.utils.write_trans_clean(path)

    datapaths.extend([Path(Config.datapath) / "dev-clean-rep",
                      Path(Config.datapath) / "dev-clean-no-rep",
                      Path(Config.datapath) / "val-clean-rep",
                      Path(Config.datapath) / "val-clean-no-rep",
                      Path(Config.datapath) / "test-clean-rep",
                      Path(Config.datapath) / "test-clean-no-rep",
                      Path(Config.datapath) / "train-clean-rep",
                      Path(Config.datapath) / "train-clean-no-rep",
                      ])

    for path in datapaths:
        data.utils.write_trans_from_source(path / "trans.csv",
                                           target_trans=path / "trans-st.csv",
                                           trans_fn=data.utils.add_speaker_start)
        data.utils.write_trans_from_source(path / "trans-st.csv",
                                           target_trans=path / "trans-id.csv",
                                           trans_fn=data.utils.add_speaker_ids)
        data.utils.write_trans_from_source(path / "trans-st.csv",
                                           target_trans=path / "trans-st-id.csv",
                                           trans_fn=lambda x: data.utils.add_speaker_ids(x, True))


@click.command()
@click.option("--merge", default=False)
@click.option("--transcribe", default=False)
@click.option("--create_vocabs", default=False)
def main(merge: bool, transcribe: bool, create_vocabs: bool):
    # Merge datasets
    if merge:
        for dataset, dataset_str, num_samples in [
            (datasets.clean_datasets["dev-clean"], "dev-clean", 10e6),
            (datasets.clean_datasets["test-clean"], "test-clean", 10e6),
            (datasets.clean_datasets["val-clean"], "val-clean", 10e6),
            (datasets.clean_datasets["train-clean-100"],
             "train-clean-100", 10e6)
        ]:
            merge_dataset(dataset, dataset_str, num_samples)

    # Write transcription files for the clean datasets
    if transcribe:
        transcribe_libri_clean()

    # Create vocab file
    if create_vocabs:
        data.utils.write_speaker_id_vocab(dataset=datasets.clean_datasets["train-clean-100"],
                                          spid_vocab_path="src/models/vocab_spid.json",)


if __name__ == "__main__":
    random.seed(Config.seed)
    main()
