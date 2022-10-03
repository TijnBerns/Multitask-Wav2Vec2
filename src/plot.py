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


@click.command()
@click.option("--jobid", default="000000", type=str, help="jobid for which plots are generated")
def main(jobid: str):
    project_dir = '.'
    root = Path(project_dir) / "logs" / "plots" / jobid
    if not root.exists():
        root.mkdir()

    # Load the measures
    train_measures = Path(project_dir) / "logs" / \
        "measures" / (jobid + ".train.json")
    val_measures = Path(project_dir) / "logs" / \
        "measures" / (jobid + ".val.json")
    train_df = read_logs(train_measures)
    val_df = read_logs(val_measures)

    # Create the plots of every column
    for column in train_df.columns:
        save_path = root / f"{jobid}_{column}.png"
        line_plot(train_df[column], val_df[column], save_path=save_path,
                  title=f"{column}", xlabel="epoch", ylabel=f"{column}")

    num_columns = len(train_df.columns)
    for i, column in enumerate(train_df.columns[:num_columns // 2]):
        column_no_sep = train_df.columns[num_columns // 2 + i]

        save_path = root / f"{jobid}_diff_{column}.png"

        x1 = train_df[column] - train_df[column_no_sep]
        x2 = val_df[column] - val_df[column_no_sep]
        line_plot(x1, x2, save_path=save_path,
                  title=f"Difference in {column}", xlabel="epoch", ylabel=f"Diff. {column}")


if __name__ == "__main__":
    main()
    # df = read_logs(Path("/home/tberns/Speaker_Change_Recognition/logs/measures/2385008.train.json"))
