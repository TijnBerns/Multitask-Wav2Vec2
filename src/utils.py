from config import Config
import os
import torch
from typing import List, Dict, Any, Tuple, Union
import json
import pandas as pd
from pathlib import Path


def json_dump(path: Union[str, Path], data: Any) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def write_dict_list(path: str, data: List[Dict]) -> None:
    with open(path, 'w') as f:
        f.write(','.join(data[0].keys()) + '\n')
        for entry in data:
            f.write(','.join(entry.values()) + '\n')
        
        
    
    # df = pd.DataFrame(data)
    # df.to_csv(path)
    return


def set_device() -> Tuple[str, str]:
    default_jobid = "0000000"
    jobid = os.environ.get("SLURM_JOB_ID", default_jobid)
    device = 'cuda' if torch.cuda.is_available and jobid != default_jobid else 'cpu'

    if jobid == default_jobid and device != "cpu":
        exit("Running on GPU without use of slurm!")
    elif jobid != default_jobid and device == "cpu":
        exit("Running slurm job without using GPU!")

    return device, jobid
