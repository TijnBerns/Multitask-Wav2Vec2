from dataclasses import dataclass


@dataclass
class Config():
    def __init__(self, job_id) -> None:
        self.job_id = job_id

    sample_rate: int = 16_000
    seed: int = 0

    # parameters for dataset and dataloaders
    batch_size: int = 1
    effective_batch_size: int = 8
    num_workers: int = 3
    max_token_count: int = 40 * sample_rate
    datapath: str = "/scratch/tberns/asr/data"
    logpath: str = "/home/tberns/Speaker_Change_Recognition/logs"
    # datapath = "/ceph/csedu-scratch/course/IMC030_MLIP/users/tberns/asr"
    # datapath: str = "/home/tijn/CS/Master/Speaker_Change_Recognition/data"
    
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
    