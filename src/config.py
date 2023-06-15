#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Config():
    def __init__(self, job_id) -> None:
        self.job_id = job_id

    # Change the following two attributes to the path where data is stored or downloaded, and logs are saved to.
    datapath: str = "/mnt/data00/audio/tijn"
    logpath: str = "./logs"

    sample_rate: int = 16_000
    seed: int = 0

    # parameters for dataset and dataloaders
    batch_size: int = 1
    effective_batch_size: int = 8
    num_workers: int = 2
    max_token_count: int = 40 * sample_rate


    # Training parameters
    lr_stage_one: float = 1e-3
    lr_stage_two: float = 3e-5
    num_steps_stage_one: int = 0
    num_steps_stage_two: int = 100_000
    # num_epochs = 20
    # num_epochs_stage_one = 0
    # num_epochs_stage_two = num_epochs - num_epochs_stage_one
    # assert num_epochs_stage_two >= 0

    # Parameters for merging samples
    speaker_change_symbol: str = '#'
    num_samples: int = 1000
    min_tokens: int = int(sample_rate * 17.5)
    max_tokens: int = int(sample_rate * 30)
    max_attempts: int = 100
    train_split: float = 0.8

