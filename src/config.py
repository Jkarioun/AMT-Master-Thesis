import librosa
import tensorflow as tf
import librosa.display
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from PIL import Image
import math
from zipfile import ZipFile
import os
import io
import soundfile as sf
import logging


logging.basicConfig(filename='../logs/trace.log',level=logging.DEBUG,\
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

HARMONIC_RELATIVES = np.array([1/3, 1/2, 1, 2, 3])
CONV_SIZE = len(HARMONIC_RELATIVES)


# debug parameters
TRAINING = False
TRAIN_FROM_LAST = True
super_path = "../tmp/dummy.ckpt"
RANDOM_DEBUG = 5000
show_images = True

# Paths
PATH_DEBUG = "../data/debug/"
PATH_PREDICT = "../data/predict/"
PATH_TEST = "../data/test/"
PATH_TRAIN = "../data/train/"
PATH_VISUALISATION = "../data/visualisation/"
PATH_MAPS = "../data/MAPS/"
TRAIN_PATHS = []
TEST_PATHS = []

# Model parameters
DEFAULT_HPARAMS = tf.contrib.training.HParams(learning_rate=0.0001)
