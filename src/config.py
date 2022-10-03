from dataclasses import dataclass


@dataclass
class Config():
    def __init__(self, job_id) -> None:
        self.job_id = job_id
        
    sample_rate: int = 16_000
    seed: int = 2022
    
    # Training parameters
    lr: float = 1e-3
    num_epochs = 20
    num_epochs_stage_one = 1
    num_epochs_stage_two = num_epochs - num_epochs_stage_one
    assert num_epochs_stage_two > 0 
         
    # parameters for dataset and dataloaders
    batch_size: int = 1
    num_workers: int = 1
    datapath: str = "/scratch/tberns/asr/data"
    logpath: str = "/home/tberns/Speaker_Change_Recognition/logs"
    # datapath = "/ceph/csedu-scratch/course/IMC030_MLIP/users/tberns/asr"
    # datapath: str = "/home/tijn/CS/Master/Speaker_Change_Recognition/data"

    # Parameters for merging samples
    speaker_change_symbol: str = '#'
    num_samples: int = 1000
    max_tokens: int = sample_rate * 15
    max_attempts: int = 100
    
    