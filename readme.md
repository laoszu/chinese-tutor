# Chinese Tutor
This is a project for classifying Mandarin Chinese（官话）tones from audio recordings, designed to assist language learners in mastering tonal pronunciation.

## Installation

```bash
pip install librosa tensorflow scikit-learn matplotlib seaborn sounddevice pydub
```

# Dataset Preparation
Download [THCHS-30 from OpenSLR](https://www.openslr.org/18/).

Extract files to `data_thchs30/` directory:

```
data_thchs30/
├── data/
│   ├── *.wav
│   └── *.wav.trn
├── train/
│   ├── *.wav
│   └── *.wav.trn
├── test/
│   ├── *.wav
│   └── *.wav.trn
├── lm_word/
│   └── lexicon.txt
└── lm_phone/
    └── lexicon.txt

```