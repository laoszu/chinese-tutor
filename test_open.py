import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sounddevice as sd

"""
    Opens an audio file and displays its transcription in the console.
    This is another testing script.
"""

def get_transcript_path(raw_path):
    try:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"File {raw_path} doesn't exist.")

        broken_path = open(raw_path).readline().rstrip()
        base_dir = os.path.dirname(raw_path)
        fixed_path = os.path.normpath(os.path.join(base_dir, broken_path))

        if not os.path.exists(fixed_path):
            raise FileNotFoundError(f"File {fixed_path} doesn't exist.")
        
        return fixed_path
    except Exception as e:
        print(f"Couldn't process file {raw_path}: {str(e)}")
        return None
    
def read_transcript(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} doesn't exist.")
        
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            print(line)

    except Exception as e:
        print(f"Couldn't process file {path}: {str(e)}")
        return None

def play_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        sd.play(audio, sr)
        sd.wait()
        print(f"Audio from: {audio_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

example_audio = "data_thchs30/train/A11_0.wav"
example_transcript = example_audio + ".trn"

play_audio(example_audio)
read_transcript(get_transcript_path(example_transcript))