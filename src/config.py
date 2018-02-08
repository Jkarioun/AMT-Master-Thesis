import librosa
import tensorflow as tf
import librosa.display
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from PIL import Image

MIDI_PITCHES = 0
NUM_PITCHES = MIDI_PITCHES + 3
BINS_PER_PITCH = 3
BINS_PER_OCTAVE = BINS_PER_PITCH * 12
TOTAL_BIN = NUM_PITCHES * BINS_PER_PITCH
MIN_FREQ = librosa.note_to_hz('A0') * (2 ** (-1 / BINS_PER_OCTAVE))
SAMPLE_RESOLUTION = 16000

CQT_WINDOW = 'hann'
CQT_HOP_LENGTH = 512

FRAME_PER_SEC = SAMPLE_RESOLUTION / CQT_HOP_LENGTH

HARMONIC_RELATIVES = np.array([-19, -12, 0, 12, 19])
# array to reorder the neurons for the harmonic convolutional layer
HARMONIC_MAPPING = np.empty(TOTAL_BIN * HARMONIC_RELATIVES.size)

# Paths
PATH_DEBUG = "../data/debug/"
PATH_PREDICT = "../data/predict/"
PATH_TEST = "../data/test/"
PATH_TRAIN = "../data/train/"
PATH_VISUALISATION = "../data/visualisation/"
