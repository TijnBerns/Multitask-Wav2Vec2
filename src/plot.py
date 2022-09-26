from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import click


def read_logs(log_file: Path):
    # Open and load json file to pandas dataframe
    with open(log_file) as f:
        data = json.load(f)

    return pd.DataFrame.from_records(data)


def line_plot(x1: list, x2: list, save_path: Path, title: str = None, xlabel: str = None, ylabel: str = None):
    # Create plots
    plt.figure(figsize=(7, 7))
    plt.plot(x1, label="Train")
    plt.plot(x2, label="Validation")

    # Set title and labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Save plot
    plt.savefig(save_path)


def main(jobid: str, project_dir: str = '.'):
    # Load the measures
    train_measures = Path(project_dir) / "logs" / \
        "measures" / (jobid + ".train.json")
    val_measures = Path(project_dir) / "logs" / \
        "measures" / (jobid + ".val.json")
    train_df = read_logs(train_measures)
    val_df = read_logs(val_measures)

    # Create the plots of every column
    for column in train_df.columns:
        save_path = Path(project_dir) / "logs" / \
            "plots" / f"{jobid}_{column}.png"
        line_plot(train_df[column], val_df[column], save_path=save_path,
                  title=f"Train {column}", xlabel="epoch", ylabel=f"{column}")


if __name__ == "__main__":
    main("2385008")
    # df = read_logs(Path("/home/tberns/Speaker_Change_Recognition/logs/measures/2385008.train.json"))
