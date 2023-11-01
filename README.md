# Multi-task Wav2Vec2

## Abstract

This research explores the diversity of the Wav2Vec2 network by
fine-tuning it to perform three distinctive speech related tasks. The
tasks we consider are automatic speech recognition (ASR), speaker-
change detection, and speaker recognition. Our approach relies on
the introduction of speaker-change tokens or speaker-identity tokens 
to the target transcriptions during network fine-tuning. In this
work, we introduce a method for extracting speaker embeddings
from the Wav2Vec2 network, and show that our approach allows
the model to perform the three distinctive tasks on an artificially
created multi-utterance dataset. Additionally, we show that the
introduction of the tokens during network fine-tuning is beneficial
for the ASR performance.

## Downloading the data and setting up the virtual environment

- First, change `datapath` in [config.py](src/config.py) to the path you wish data to be downloaded.
- Change `logpath` in [config.py](src/config.py) to the path you wish model checkpoints to be saved to.
- Run the following from the root directory of this project to create a virtual environment with all the required packages:\
```./scripts/setup_venv```
- Run the following from the root directory of this project to download and preprocess the data (this may take some time):\
```python src/data/dataprep.py --merge=True --transcribe=True --create_vocabs=True --create_trials=True```


## Training the wav2vec2 network
To train a network you can run one of the files structured as `train_tp1_single.sh` in the [scripts](./scripts/) folder.

You probably want to change the paths, according to the changes you made in [config.py](src/config.py), and the Slurm parameters, if you use Slurm.

The files are as follows:
- `train_tp1_single.sh`: Train on single utterance samples w/o speaker change tokens in the transcriptions.
- `train_tp1_multi.sh`: Train on multi utterance samples w/o speaker change tokens in the transcriptions.
- `train_tp2_single.sh`: Train on single utterance samples with speaker change tokens in the transcriptions.
- `train_tp2_multi.sh`: Train on multi utterance samples with speaker change tokens in the transcriptions.
- `train_tp2_single.sh`: Train on single utterance samples with speaker identity tokens in the transcriptions.
- `train_tp2_multi.sh`: Train on multi utterance samples with speaker identity tokens in the transcriptions.


## Evaluating the checkpoints
To evaluate a checkpoint, run the `./scripts/eval_tp1 VERSION` where `VERSION` is replaced with the version of the checkpoint, and the suffix `tp1` corresponds to the type of transcriptions that are used during evaluation:
- `tp1`: No speaker change tokens or speaker identity tokens.
- `tp2`: Speaker change tokens.
- `tp3`: Speaker identity tokens.

Pytorch lightning automatically saves model checkpoints as `lightning_logs/version_VERSION`. When running the eval script, we look for the best checkpoint in the `lightning_logs/version` directory that is automatically created when training a model.
