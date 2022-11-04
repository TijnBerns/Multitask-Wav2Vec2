
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

def line_plot(y, x=None, xlabel='', ylabel='', title='', save=None):
    if x is None:
        x = list(range(len(y)))
        
    # Create plot
    plt.figure(figsize=(3.5,3.5))
    plt.plot(x, y)
    
    # Add title and labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
        data = json.load(f)[0]
    line_plot(data['EER'], xlabel='Hidden state', ylabel='EER (%)', save=save)


if __name__ == "__main__":
    main()
    