import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

"""
    Draws a spectrogram of a randomly selected audio file.
    This script is intended for testing purposes.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("file", type=str, help="Path to audio file")

    args = parser.parse_args()

    path = args.file
    
    audio, sr = librosa.load(path, sr=None) # original probing

    D = librosa.stft(audio)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # dB are more 'fit' to our perception of hearing

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="log", cmap="viridis")

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectogram of {path}")
    plt.show()