from dataclasses import dataclass


@dataclass
class Config: 
    # parameters for dataset
    batch_size: int = 1
    num_workers: int = 1
    datapath = "/scratch/tberns/asr/data"
    # datapath = "/ceph/csedu-scratch/course/IMC030_MLIP/users/tberns/asr"
    # datapath: str = "/home/tijn/CS/Master/Speaker_Change_Recognition/data"

    sample_rate: int = 16_000
    seed = 2022
    speaker_change_symbol = ' # '
    # Version number 
    version: int = 0