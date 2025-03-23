import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import ListedColormap

def detect_speech_regions(audio_path, sr=16000):
    try:
        audio, sr = librosa.load(audio_path, sr=sr)
        audio = librosa.util.normalize(audio)

        frame_length = 4096
        hop_length = frame_length // 8

        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        silence_threshold = np.percentile(rms, 22)
        voiced_mask = rms > silence_threshold

        return {
            'times': librosa.times_like(rms, sr=sr, hop_length=hop_length),
            'audio': audio,
            'sr': sr,
            'hop_length': hop_length,
            'voiced_mask': voiced_mask
        }
    
    except Exception as e:
        print(f"Could not process the file: {e}")
        return None

def plot_audio_analysis(audio_path):
    data = detect_speech_regions(audio_path)

    if data is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), 
                        gridspec_kw={'height_ratios': [3, 1]})
    
    S = librosa.stft(data['audio'], n_fft=4096, hop_length=data['hop_length'])
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    librosa.display.specshow(S_db, x_axis='time', y_axis='log',
                            sr=data['sr'], hop_length=data['hop_length'],
                            cmap='magma', ax=ax1)
    
    # highlight energy range --- possibly a speech regions (-25dB to 0dB)
    highlight_mask = (S_db >= -25) & (S_db <= 0)

    # green color
    highlight_cmap = ListedColormap([(0,0,0,0), (0,1,0,0.5)])
    librosa.display.specshow(highlight_mask, 
                            x_axis='time', 
                            y_axis='log',
                            sr=data['sr'],
                            hop_length=data['hop_length'],
                            cmap=highlight_cmap,
                            ax=ax1)
    
    ax1.fill_between(data['times'], 70, 450, 
                    where=data['voiced_mask'][:len(data['times'])],
                    color='cyan', alpha=0.2, 
                    label='Speech regions')
    
    ax1.set_title('Spectrogram with speech regions (-25 to 0 dB)', fontsize=16, pad=15)
    ax1.set_ylim(70, 450)
    ax1.legend(loc='upper right', fontsize=10)
    
    times = np.linspace(0, len(data['audio'])/data['sr'], len(data['audio']))
    ax2.plot(times, data['audio'], alpha=0.8)
    ax2.fill_between(data['times'], -1, 1, 
                    where=data['voiced_mask'][:len(data['times'])], 
                    alpha=0.2, color='cyan',
                    label='Speech regions')
    
    ax2.set_title('Amplitude plot with speech regions', fontsize=14)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_ylim(-1, 1)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio analysis with speech region detection")
    parser.add_argument("file", type=str, help="Path to audio file")
    args = parser.parse_args()
    
    plot_audio_analysis(args.file)