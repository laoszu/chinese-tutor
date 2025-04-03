import torch
import librosa as lb
import numpy as np
from pathlib import Path
import re
from torch.utils.data import Dataset

class ToneDataset(Dataset):
    def __init__(
        self, 
        dir: str, 
        ext: str = "mp3",
        target_length: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fixed_time: int = 44
    ):
        self.dir = Path(dir)
        self.files = list(self.dir.glob(f"*.{ext}"))
        self.target_length = target_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_time = fixed_time

        if not self.files:
            raise FileNotFoundError(f"No files found in {dir}")
        
        self.labels = []
        pattern = re.compile(r"([a-zA-Z]+?\d)$")
        
        for file in self.files:
            match = pattern.search(file.stem)
            if not match:
                raise ValueError(f"Invalid filename: {file.name}")
            self.labels.append(int(match.group(1)[-1]) - 1)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        y, sr = lb.load(file_path, sr=22050, mono=True)
        
        if len(y) > self.target_length:
            y = y[:self.target_length]
        else:
            y = np.pad(y, (0, max(0, self.target_length - len(y))), mode="constant")
        
        mel = lb.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = lb.power_to_db(mel, ref=np.max)
        
        if mel_db.shape[1] < self.fixed_time:
            pad_width = self.fixed_time - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
        elif mel_db.shape[1] > self.fixed_time:
            mel_db = mel_db[:, :self.fixed_time]
        
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
        return torch.from_numpy(mel_db).float().unsqueeze(0), self.labels[idx]