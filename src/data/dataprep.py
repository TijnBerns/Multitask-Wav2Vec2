import sys
sys.path.append('src')
import click
from pathlib import Path
import random
from config import Config
from typing import List, Union, Dict
import data.datasets as datasets
import data.utils
from data.generate_merged_samples import PairGeneratorRepeat, PairGeneratorNoRepeat
from tqdm import tqdm


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
    
    for dataset_str, dataset in tqdm(datasets.clean_datasets.items(), desc="Writing transcriptions for clean datasets"):
        data.utils.write_trans_clean(dataset, dataset_str, Config.datapath + "/LibriSpeech")
        
    trans_files = Path(Config.datapath).rglob("*trans.csv")
    
    for trans_file in tqdm(trans_files, desc="Writing additional transcription files"):
        path = trans_file.parent
        name = trans_file.name[:-4]
        
        data.utils.write_trans_from_source(source_trans=trans_file,
                                           target_trans=path / f"{name}-st.csv",
                                           trans_fn=data.utils.add_speaker_start)
        data.utils.write_trans_from_source(path / f"{name}-st.csv",
                                           target_trans=path / f"{name}-id.csv",
                                           trans_fn=data.utils.add_speaker_ids)
        data.utils.write_trans_from_source(path / f"{name}-st.csv",
                                           target_trans=path / f"{name}-st-id.csv",
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
