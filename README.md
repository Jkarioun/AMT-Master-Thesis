# Deep Learning in Automatic Piano Transcription

## Description

The goal of this EPL master thesis is to create a piano-roll
representation of a wav piano music using deep neural networks.
The main part of this thesis is currently to analyse the influence of a
custom convolutional layer - the harmonic convolutional layer - which
uses harmonic frequencies instead of neighbouring ones.
The implementation of this layer is done in model.py.
We use the [Kelz](https://arxiv.org/abs/1612.05153) implementation as a
baseline, and will further improve to the [Onsets and Frames](https://arxiv.org/abs/1710.11153) model of magenta .

## Authors

This work is done in the context of a Master's thesis at the
Université catholique de Louvain by

**KARIOUN Jaâfar** and **TIHON Simon**

and supervised by

**DE VLEESCHOUWER Christophe** and **GANSEMAN Joachim**.

# Running the project

## Dependencies
The project target python3.5 or 3.6 and depends on tensorflow, librosa,
pretty-midi, image, soundfile, matplotlib, sklearn.

To install the libraries, run the following commands:

```bash
pip3 install --upgrade tensorflow
pip3 install librosa pretty-midi image soundfile matplotlib sklearn
```


## Dataset

The dataset used is the [MAPS dataset](www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/)
[1]. It should be placed in the data/MAPS directory.

The program need the dataset to be preprocessed, which is done by running
the following command from the src/ directory:

```bash
python3 preprocessing.py
```

## Main program

Running the program is done by running the following command from the
src/ directory after setting the running parameters in src/config.py:

```bash
python3 main.py
```

# References

[1] Multipitch estimation of piano sounds using a new probabilistic
spectral smoothness principle, V. Emiya, R. Badeau, B. David, IEEE
Transactions on Audio, Speech and Language Processing, 2010.

[2] Colin Raffel and Daniel P. W. Ellis. Intuitive Analysis, Creation and
Manipulation of MIDI Data with pretty_midi. In 15th International
Conference on Music Information Retrieval Late Breaking and Demo Papers, 2014.

[3] **Hawthorne et al.(2017)**, *Onsets and Frames: Dual-Objective Piano Transcription*,
Hawthorne, C., Elsen, E., Song, J., et al., 2017, arXiv:1710.11153

