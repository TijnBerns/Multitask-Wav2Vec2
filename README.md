# Multi-task Wav2Vec2

## Downloading the data and setting up the virtual environment
Change `datapath` in `src/config.py` to the path you whish data to be downloaded.
Change `logpath` in `src/config.py` to the path you whish model checkpoints to be saved to.

Run the following from the root directory of this project to create a virtual environment with all the required packages:\
```./scripts/setup_venv``` 

Run the following from the root directory of this project to download and preprocess the data (this may take some time):\
```python src/data/dataprep.py --merge=True --transcribe=True --create_vocabs=True```



## Training the wav2vec2 network
To train a network you can run one of the files structured as `train_tp1_single.sh` in the `scripts` folder.
The files are as follows:
- `train_tp1_single.sh`: Train on single utterance samples w/o speaker change tokens in the transcriptions.
- `train_tp1_multi.sh`: Train on multi utterance samples w/o speaker change tokens in the transcriptions.
- `train_tp2_single.sh`: Train on single utterance samples with speaker change tokens in the transcriptions.
- `train_tp2_multi.sh`: Train on multi utterance samples with speaker change tokens in the transcriptions.
- `train_tp2_single.sh`: Train on single utterance samples with speaker idenity tokens in the transcriptions.
- `train_tp2_multi.sh`: Train on multi utterance samples with speaker idenity tokens in the transcriptions.


## Evaluating the checkpoints
To evaluate a checkpoint, run the `./scripts/setup_venv VERSION` where `VERSION` is replaced with the version of the checkpoint.
Pytorch lightning automatically saves model checkpoints as `version_VERSION`. When running the eval script, we look for checkpoints in the `lightning_logs` 
directory that is automatically created when training a model.