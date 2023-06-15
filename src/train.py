#!/usr/bin/env python3

from config import Config
import data.datasets as data
import utils
import models.wav2vec2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import click
from typing import Union, List
from tqdm import tqdm


@click.command()
@click.option("--train_trans", multiple=True,
              help="List or string of path(s) in which train transcriptions are stored.")
@click.option("--val_trans", multiple=True,
              help="List or string of path(s) in which validation transcriptions are stored.")
@click.option("--vocab_path", default=None,
              help="Path to the model vocab file.")
def main(train_trans: Union[List[str], str, None], val_trans: Union[List[str], str, None], vocab_path: str,):
    pl.seed_everything(Config.seed)
    device, _ = utils.set_device()
    train_trans = list(train_trans)
    val_trans = list(val_trans)

    # Load datasets
    train_pipe = data.build_datapipe(train_trans, dynamic_batch_size=False)
    val_pipe = data.build_datapipe(val_trans, dynamic_batch_size=False)

    # Initialize dataloaders
    train_loader = data.initialize_loader(train_pipe, shuffle=True)
    val_loader = data.initialize_loader(val_pipe, shuffle=False)

    # Initialize checkpointer
    pattern = "epoch_{epoch:04d}.step_{step:09d}.val-wer_{val_wer:.4f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        save_top_k=1,
        # every_n_epochs=5,
        every_n_train_steps=500,
        monitor="val_wer",
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Load wav2vec2 module
    wav2vec2_module = models.wav2vec2.Wav2Vec2Module(num_steps_stage_one=Config.num_steps_stage_one,
                                                     num_steps_stage_two=Config.num_steps_stage_two,
                                                     lr_stage_one=Config.lr_stage_one,
                                                     lr_stage_two=Config.lr_stage_two,
                                                     batch_size=Config.batch_size,
                                                     stage=1,
                                                     vocab_path=vocab_path)


    wav2vec2_module = wav2vec2_module.to(device)

    # First stage
    wav2vec2_module.freeze_all_but_head()
    first_stage = pl.Trainer(max_steps=Config.num_steps_stage_one,
                             accelerator=device,
                             callbacks=[checkpointer],
                             log_every_n_steps=50,
                             accumulate_grad_batches=Config.effective_batch_size
                             )
    first_stage.fit(model=wav2vec2_module,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

    # Prepare second stage
    wav2vec2_module.unfreeze()
    wav2vec2_module.model.freeze_feature_encoder()
    wav2vec2_module.stage = 2
    wav2vec2_module.configure_optimizers()

    # Second stage
    lr_monitor = LearningRateMonitor(logging_interval='step')
    second_stage = pl.Trainer(max_steps=Config.num_steps_stage_two,
                              accelerator=device,
                              callbacks=[checkpointer, lr_monitor],
                              log_every_n_steps=5,
                              accumulate_grad_batches=Config.effective_batch_size
                              )
    second_stage.fit(model=wav2vec2_module,
                     train_dataloaders=train_loader,
                     val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
