from config import Config
import data
import utils
import models.wav2vec2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def main(device: str, jobid: str):
    # Load datasets
    train_set = data.CustomLibriSpeechDataset([
        Config.datapath + '/train-clean-no-rep',
        Config.datapath + '/train-clean-rep'])
    val_set = data.CustomLibriSpeechDataset([
        Config.datapath + '/val-clean-no-rep',
        Config.datapath + '/val-clean-rep'])

    # Initialize dataloaders
    train_loader = data.initialize_loader(train_set, shuffle=True)
    val_loader = data.initialize_loader(val_set, shuffle=False)

    # Initialize checkpointer
    pattern = "epoch_{epoch:04d}.step_{step:09d}.val-wer_{val_wer:.4f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        # save_top_k=-1,
        # every_n_epochs=5,
        monitor="val_wer",
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Load wav2vec2 module
    wav2vec2_module = models.wav2vec2.Wav2Vec2Module(num_epochs=Config.num_epochs,
                                                     lr_stage_one=Config.lr_stage_one,
                                                     lr_stage_two=Config.lr_stage_two,
                                                     batch_size=Config.batch_size,
                                                     stage=1)
    wav2vec2_module = wav2vec2_module.to(device)

    # First stage
    wav2vec2_module.freeze_all_but_head()
    first_stage = pl.Trainer(max_epochs=Config.num_epochs_stage_one,
                             accelerator=device,
                             callbacks=[checkpointer],
                             log_every_n_steps=200)
    first_stage.fit(model=wav2vec2_module,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

    # Second stage
    wav2vec2_module.unfreeze()
    wav2vec2_module.model.freeze_feature_encoder()
    wav2vec2_module.stage = 2
    wav2vec2_module.configure_optimizers()
    second_stage = pl.Trainer(max_epochs=Config.num_epochs_stage_two,
                              accelerator=device,
                              callbacks=[checkpointer],
                              log_every_n_steps=200)
    second_stage.fit(model=wav2vec2_module,
                     train_dataloaders=train_loader,
                     val_dataloaders=val_loader)


if __name__ == "__main__":
    device, jobid = utils.set_device()
    main(device, jobid)
