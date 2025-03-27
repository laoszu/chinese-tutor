import sys
import textwrap
import librosa
import matplotlib
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import ListedColormap
from test_open import get_transcript_path, read_transcript
from pydub import AudioSegment,silence

"""
    Naive speech recognition.
    Provided with an audio path, displays:
        the waveform
        amplitude spectogram
        deciBel spectogram with marked speech regions (cyan) and "tones' scrapes" (green) - areas where the energy is accumulated

    Run it:
        python detect_speech.py <path>
"""

SR = 16000
FFT_WINDOW = 2048
HOP_LENGTH = FFT_WINDOW // 8
SILENCE_THRESHOLD = 30

def detect_speech_regions(audio_path, sr=None):
    try:
        audio, sr = librosa.load(audio_path, sr=sr)
        audio = librosa.util.normalize(audio)

        rms = librosa.feature.rms(y=audio, frame_length=FFT_WINDOW, hop_length=HOP_LENGTH)[0]
                
        # PART of audio range is selected, the rest is treated as silence (xx% is silence threshold)
        silence_threshold = np.percentile(rms, SILENCE_THRESHOLD)
        mask = rms > silence_threshold

        return {
            'times': librosa.times_like(rms, sr=sr, hop_length=HOP_LENGTH),
            'audio': audio,
            'mask': mask
        }
    
    except Exception as e:
        print(f"Could not process the file: {e}")
        return None

def plot_audio_analysis(audio_path):
    data = detect_speech_regions(audio_path)

    if data is None:
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 10))
    
    # pinyin transcript in the top
    trn_path = get_transcript_path(audio_path)
    if trn_path != None:
        text = read_transcript(trn_path)[1]
        wrapped_text = textwrap.fill(text, width=80)
        plt.suptitle(wrapped_text, fontsize=16, ha='center', va='top')

    # waveform
    times = np.linspace(0, len(data['audio'])/SR, len(data['audio']))
    ax1.plot(times, data['audio'], alpha=0.8)
    ax1.fill_between(data['times'], -1, 1, 
                    where=data['mask'][:len(data['times'])], 
                    alpha=0.2,
                    label='Speech regions')
    
    ax1.set_title('waveform with speech regions', fontsize=14)
    ax1.set_ylim(-1, 1)

    # amplitude scaled spectogram (raw FFT)
    S = librosa.stft(data['audio'], n_fft=FFT_WINDOW, hop_length=HOP_LENGTH)
    librosa.display.specshow(S, x_axis='time', y_axis='log',
                            sr=SR, hop_length=HOP_LENGTH,
                            cmap='magma', ax=ax2)
    
    ax2.set_title('Spectogram', fontsize=16, pad=15)
    ax2.set_ylim(70, 450)
    
    # mel spectogram (more similar to human perception of hearing)
    magnitude, phase = librosa.magphase(S)
    mel_scale_sgram = librosa.feature.melspectrogram(S=magnitude, sr=SR)
    S_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

    librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                            sr=SR, hop_length=HOP_LENGTH,
                            cmap='magma', ax=ax3)
    
    ax3.fill_between(data['times'], 70, 450, 
                    where=data['mask'][:len(data['times'])],
                    color='cyan', alpha=0.2, 
                    label='Speech regions')
    
    ax3.set_title('Mel spectogram', fontsize=16, pad=15)
    ax3.set_ylim(70, 450)
    legend_labels = [
        Patch(color='cyan', alpha=0.2, label='Speech regions'),
        Patch(color='green', alpha=0.5, label='Energy range (-25, 0) dB')
    ]

    ax3.legend(handles=legend_labels, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Audio analysis with speech region detection")
    parser.add_argument("file", type=str, help="Path to audio file")
    args = parser.parse_args()
    plot_audio_analysis(args.file)