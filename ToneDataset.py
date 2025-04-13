import torch
import librosa as lb
import numpy as np
from pathlib import Path
import re
from torch.utils.data import Dataset

class ToneDataset(Dataset):
    def __init__(self, dir, target_length=22050, n_mels=128, n_fft=2048, hop_length=512, fixed_time=44):
        '''
            dir: where the dataset is placed
            target_length: audio samples for each signal
            n_mels: mel bands in mel-spect, resolution of the frequency axis in the spect
            n_fft: samples in fft window
            hop_length: samples between each fft window - step size
            fixed_time: time frames (columns) in the spect
        '''

        self.dir = Path(dir)
        self.files = list(self.dir.glob("*.mp3"))
        self.target_length = target_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_time = fixed_time
        self.filenames = [f.name for f in self.files]
        
        if not self.files:
            raise FileNotFoundError(f"No files found in {dir}")
        
        self.labels = []
        pattern = re.compile(r"([a-zA-Z]+)(\d)")    # get first digit (tone class) and the first word (label)
        
        for file in self.files:
            match = pattern.search(file.stem)
            if not match:
                raise ValueError(f"Invalid filename: {file.name}")
            tone_number = int(match.group(2))
            if tone_number < 1 or tone_number > 4:
                raise ValueError(f"Invalid tone number in filename: {file.name} (tone: {tone_number})")
            self.labels.append(tone_number - 1)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        y, sr = lb.load(file_path, sr=22050, mono=True)
        
        # trim/pad the audio to the target length
        if len(y) > self.target_length:
            y = y[:self.target_length]
        else:
            y = np.pad(y, (0, max(0, self.target_length - len(y))), mode="constant")
        
        # to the mel spectogram (and change the scale to db)
        mel = lb.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, 
                                        hop_length=self.hop_length, n_mels=self.n_mels)
        mel_db = lb.power_to_db(mel, ref=np.max)
        
        # spect time dimension to fixed_time columns
        if mel_db.shape[1] < self.fixed_time:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.fixed_time - mel_db.shape[1])), mode="constant")
        elif mel_db.shape[1] > self.fixed_time:
            mel_db = mel_db[:, :self.fixed_time]
        
        # normalize the spectrogram (0, 1)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
        return torch.from_numpy(mel_db).float().unsqueeze(0), self.labels[idx]
    
    def __len__(self):
        return len(self.files)