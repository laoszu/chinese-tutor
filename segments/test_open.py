import argparse
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
            while line := f.readline():
                print(line.rstrip())

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

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to audio file")
    args = parser.parse_args()

    transcript = args.file + ".trn"

    play_audio(args.file)
    read_transcript(get_transcript_path(transcript))