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
    │   └── ...
    ├── dev/
    │   └── ...
    ├── test/
    │   └── ...
    ├── lm_word/
    │   ├── word.3gram.lm
    │   └── lexicon.txt
    └── lm_phone/
        ├── phone.3gram.lm
        └── lexicon.txt
```

To find out more about this dataset, [check out my analysis](data_thchs30.ipynb)!

# Credits (thchs30)
Authors:
- Dong Wang
- Xuewei Zhang
- Zhiyong Zhang

Contactor:
- Dong Wang, Xuewei Zhang, Zhiyong Zhang
- wangdong99@mails.tsinghua.edu.cn
- zxw@cslt.riit.tsinghua.edu.cn
- zhangzy@cslt.riit.tsinghua.edu.cn

CSLT, Tsinghua University

ROOM1-303, BLDG FIT <br/>
Tsinghua University

http://cslt.org <br/>
http://cslt.riit.tsinghua.edu.cn