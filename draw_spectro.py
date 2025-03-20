import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

"""
    Draw spectogram of a randomly selected file.
    Script with testing purpose.
"""
audio, sr = librosa.load("data_thchs30/train/A11_0.wav", sr=None) # original probing

D = librosa.stft(audio)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # dB are more 'fit' to our perception of hearing

plt.figure(figsize=(12, 6))
librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="log", cmap="viridis")

plt.colorbar(format="%+2.0f dB")
plt.title("Spectogram of A11_0.wav")
plt.show()