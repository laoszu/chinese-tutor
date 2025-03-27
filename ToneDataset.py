import torch
import librosa as lb
from pathlib import Path
import re
from torch.utils.data import Dataset

class ToneDataset(Dataset):
    def __init__(self, dir, ext="mp3"):
        self.dir = Path(dir)
        self.files = list(self.dir.glob(f"*.{ext}"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No suitable files found in {dir}")
        
        self.label_strs = []
        pattern = re.compile(r"([a-zA-Z]+?\d)")       # any letter or digit

        for file in self.files:
            match = pattern.search(file.stem)
            if not match:
                raise ValueError(f"Invalid filename pattern in {file.name}")
            self.label_strs.append(match.group(1))
            
        self.labels = [label for label in self.label_strs]

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform, _ = lb.load(file_path, sr=22050, mono=True)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        label = self.labels[idx]
        
        return waveform, label[:-1], int(label[-1])
    
    def __len__(self):
        return len(self.files)