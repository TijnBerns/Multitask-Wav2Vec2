from dataclasses import dataclass


@dataclass
class Config():
    def __init__(self, job_id) -> None:
        self.job_id = job_id
        
    sample_rate: int = 16_000
    seed = 2022
    
         
    # parameters for dataset and dataloaders
    batch_size: int = 2
    num_workers: int = 1
    datapath = "/scratch/tberns/asr/data"
    # datapath = "/ceph/csedu-scratch/course/IMC030_MLIP/users/tberns/asr"
    # datapath: str = "/home/tijn/CS/Master/Speaker_Change_Recognition/data"

    
    
    
    # Parameters for merging samples
    speaker_change_symbol = '#'
    num_samples = 1000
    max_tokens = sample_rate * 15
    max_attempts = 100
    
    