################################################################################
#
# Implement an Evaluator object which encapsulates the process
# computing performance metric of speaker recognition task.
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any
from warnings import warn

import numpy as np
import torch as t

from torch.nn.functional import normalize
from torch.nn import CosineSimilarity
from tqdm import tqdm

from evaluation.metrics import calculate_eer

from pathlib import Path


################################################################################
# define data structures required for evaluating


@dataclass
class EmbeddingSample:
    sample_id: str
    index: int
    embedding: t.Tensor
    
    def __call__(self, *args: Any, **kwds: Any) -> t.Tensor:
        return self.embedding
    
    
if __name__ == "__main__":
    pass 
    

