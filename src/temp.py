import fileinput
from config import Config
from pathlib import Path

def convert_to_lower(filename):
    for line in fileinput.input(filename, inplace=1):
        newline = line.lower()
        newline = newline.replace('librispeech', "LibriSpeech")
        print(newline, end='')
        
        
if __name__ == "__main__":
    trans_files = Path(Config.datapath).rglob("*trans*.csv")
    for path in trans_files:
        print(path)
        convert_to_lower(path) 