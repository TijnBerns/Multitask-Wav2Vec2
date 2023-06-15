#!/usr/bin/env python3

import json
import click
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def line_plot(data, xlabel='', ylabel='', title='', save=None):
    x = None

    dev_eer = list(data["dev"].values())[0]
    test_eer = list(data["test"].values())[0]

    if x is None:
        x = list(range(len(dev_eer)))

    # Create plot
    plt.figure(figsize=(4,3))
    plt.plot(x, dev_eer, label="dev")
    plt.plot(x, test_eer, label="test")

    # Add title and labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

    # Save the figure
    if save is None:
        return
    plt.savefig(save + '.png', bbox_inches = "tight")
    plt.savefig(save + '.pgf', bbox_inches = "tight")

@click.command()
@click.option('--file', type=Path)
# @click.option('--save', type=str)
def main(file: Path):
    version_number = file.name.split('-')[0]
    save = f'logs/plots/{version_number}-eer'
    with open(file, 'r') as f:
        data = json.load(f)
    line_plot(data, xlabel='Encoder block', ylabel='\\textit{eer} (\%)', save="eer")


if __name__ == "__main__":
    main()
