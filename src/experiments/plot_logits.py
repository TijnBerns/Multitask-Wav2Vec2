import click
import utils
import data.data as data
from config import Config

from pathlib import Path
import models.wav2vec2_spch
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from typing import List

torch.set_printoptions(profile="full")


def dist_matrix(logits: torch.Tensor) -> torch.Tensor:
    res = torch.zeros((len(logits), 33, 33))

    for i, logit in enumerate(logits):
        res[i] = torch.abs((logit * torch.ones((33, 33))).T -
                           logit * torch.ones((33, 33)))
    return torch.mean(res, dim=0)


def normalize_matrix(matrix):
    return torch.nn.functional.normalize(matrix.flatten(), dim=0).view((*matrix.size(),))


def plot_logit_matrix(data2d: List[List[float]], path: str, title: str) -> None:
    with open("src/models/vocab.json", "r") as f:
        vocab: dict = json.load(f)

    _ = plt.figure(figsize=(9, 9))
    im = plt.imshow(data2d)
    plt.xticks(range(33), list(vocab.keys()), rotation=60)
    plt.yticks(range(33), list(vocab.keys()))
    plt.title(title)
    plt.colorbar(im)
    plt.savefig(path)
    return


@click.command()
@click.option("--checkpoint_path", default="/home/tberns/Speaker_Change_Recognition/lightning_logs/version_2394591/checkpoints/epoch_0018.step_000364363.val-loss_0.0319.last.ckpt", help="Path to model checkpoint. If None, finetuned wav2vec2 model is used.")
def main(checkpoint_path: str = None):
    device, _ = utils.set_device()
    wav2vec2_module: models.wav2vec2_spch.Wav2Vec2Module = models.wav2vec2_spch.Wav2Vec2Module.load_from_checkpoint(
        checkpoint_path)
    ckpt_version = int(Path(checkpoint_path).parts[-3][-7:])

    # Model
    wav2vec2_module.eval()
    wav2vec2_module = wav2vec2_module.to(device)

    # Dataset
    dataset = data.CustomLibriSpeechDataset(
        [Config.datapath + '/dev-clean-no-rep'])
    dataloader = data.initialize_loader(dataset, shuffle=False)

    # Variables used saving matrix and taking means
    vocab_size = 33
    token_logits = torch.zeros((vocab_size, vocab_size, vocab_size))
    all_logits = torch.zeros((vocab_size, vocab_size))
    hit_counts = [0] * vocab_size
    token_count = 0

    for batch in tqdm(dataloader):
        waveform = batch['waveform']
        transcription = batch['transcription']

        # Forward pass
        with torch.no_grad():
            output, _ = wav2vec2_module._train_val_step(
                waveform, transcription)
        logits = output.logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Update logits distance matrices for all tokens
        for token in range(vocab_size):
            predicted_token = logits[torch.where(predicted_ids == token)]
            if len(predicted_token) != 0:
                token_logits[token] += dist_matrix(predicted_token)
                hit_counts[token] += len(predicted_token)

        # Update counts
        all_logits += dist_matrix(logits[0])
        token_count += len(logits[0])

    # Take mean and plot logit matrix for all tokens
    for token in range(vocab_size):
        # token_logits[token] = token_logits[token] / max(1, hit_counts[token])
        token_logits[token] = normalize_matrix(token_logits[token])
        plot_logit_matrix(token_logits[token], f"token-{token}-version-{ckpt_version}.png", f"argmax = {token}")
    
    # all_logits = all_logits / token_count
    all_logits = normalize_matrix(all_logits)
    plot_logit_matrix(all_logits, f"logits-version-{ckpt_version}.png", "all logits")


if __name__ == "__main__":
    main()
