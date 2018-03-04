import librosa
import tensorflow as tf
import numpy as np
import pretty_midi as pm
import tensorflow.contrib.slim as slim
from PIL import Image
import math
from zipfile import ZipFile
import os
import io
import soundfile as sf
import logging
from strings import *

DISPLAY = False

if DISPLAY:
    import librosa.display
    import matplotlib.pyplot as plt

CONFIG_NAME = 'onsets_accuracy'

logging.basicConfig(filename='../logs/%s.log' % CONFIG_NAME, level=logging.DEBUG, \
                    format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

PIANO_MIN_PITCH = 21
PIANO_MAX_PITCH = 108
ANALYSIS_UPPER_PITCH = 0  # supplementary pitches for harmonics
PIANO_PITCHES = PIANO_MAX_PITCH - PIANO_MIN_PITCH + 1
MIDI_PITCHES = 128
NUM_PITCHES = PIANO_PITCHES + ANALYSIS_UPPER_PITCH
BINS_PER_PITCH = 4
BINS_PER_OCTAVE = BINS_PER_PITCH * 12
TOTAL_BIN = NUM_PITCHES * BINS_PER_PITCH
MIN_FREQ = librosa.note_to_hz('A0') * (2 ** (-1 / BINS_PER_OCTAVE))
SAMPLE_RESOLUTION = 16000

CQT_WINDOW = 'hann'
CQT_HOP_LENGTH = 512

FRAME_PER_SEC = SAMPLE_RESOLUTION / CQT_HOP_LENGTH

HARMONIC_RELATIVES = np.array([1 / 3, 1 / 2, 1, 2, 3])
CONV_SIZE = len(HARMONIC_RELATIVES)


TOO_LONG_MIN = FRAME_PER_SEC * 2
TOO_LONG_MAX = FRAME_PER_SEC * 5

# debug parameters
ONSET = True
TRAINING = True
TRAIN_FROM_LAST = False
super_path = "../tmp/%s.ckpt" % CONFIG_NAME
show_images = False
RANDOM_DEBUG = 0
NUM_BATCHES = 100000
MIN_FRAME_PER_BATCH = 1000
MAX_FRAME_PER_BATCH = 2000

# Paths
PATH_DEBUG = "../data/debug/"
PATH_PREDICT = "../data/predict/"
PATH_TEST = "../data/test/"
PATH_TRAIN = "../data/train/"
PATH_VISUALISATION = "../data/visualisation/"
PATH_MAPS = "../data/MAPS/"
PATH_LOGS = "../logs/"
TRAIN_PATHS = []
TEST_PATHS = []

# Model parameters
DEFAULT_HPARAMS = tf.contrib.training.HParams(learning_rate=0.0002)
