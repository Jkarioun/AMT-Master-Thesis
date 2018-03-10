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
import sys
import soundfile as sf
import logging
import shutil
import errno
import time

DISPLAY = False
if DISPLAY:
    import librosa.display
    import matplotlib.pyplot as plt


PATH_SRC = "../src1"

######################
# running parameters #
######################

CONFIG_NAME = 'frame_weighted_hybrid_LR00005_UP0'
TRAINING = True
TESTING = False
TRAIN_FROM_LAST = False
show_images = False
RANDOM_DEBUG = 0
NUM_BATCHES = 50000
MIN_FRAME_PER_BATCH = 1000
MAX_FRAME_PER_BATCH = 2000


####################
# model parameters #
####################

ONSET = False
KELZ_MODEL = False
FIRST_LAYER_HARMONIC = False
HARMONIC_RELATIVES = np.array([1 / 3, 1 / 2, 1, 2, 3])
CONV_SIZE = len(HARMONIC_RELATIVES)
DEFAULT_HPARAMS = tf.contrib.training.HParams(learning_rate=0.00005)


##############################
# pitch dimension parameters #
##############################

PIANO_MIN_PITCH = 21
PIANO_MAX_PITCH = 108
ANALYSIS_UPPER_PITCH = 0  # supplementary pitches for harmonics
PIANO_PITCHES = PIANO_MAX_PITCH - PIANO_MIN_PITCH + 1
MIDI_PITCHES = 128
NUM_PITCHES = PIANO_PITCHES + ANALYSIS_UPPER_PITCH
BINS_PER_PITCH = 4
BINS_PER_OCTAVE = BINS_PER_PITCH * 12
TOTAL_BIN = NUM_PITCHES * BINS_PER_PITCH


#######################
# data CQT parameters #
#######################

MIN_FREQ = librosa.note_to_hz('A0') * (2 ** (-1 / BINS_PER_OCTAVE))
SAMPLE_RESOLUTION = 16000
CQT_WINDOW = 'hann'
CQT_HOP_LENGTH = 512
FRAME_PER_SEC = SAMPLE_RESOLUTION / CQT_HOP_LENGTH


#########################
# data other parameters #
#########################

USE_ENSTDk = False
MIN_NOTE_LENGTH_IF_SUSTAIN = FRAME_PER_SEC * 2
MAX_NOTE_LENGTH = FRAME_PER_SEC * 5


#########
# Paths #
#########

PATH_OUTPUT = "../outputs/" + CONFIG_NAME + "/"
PATH_DEBUG = "../data/debug/"
PATH_MAPS = "../data/MAPS/"
PATH_VISUALISATION = PATH_OUTPUT + "visualisation/"
PATH_TENSORBOARD = PATH_OUTPUT + "tensorboard/"
PATH_LOGS = PATH_OUTPUT + "logs/"
PATH_CHECKPOINTS = PATH_OUTPUT + "ckpt/"
PATH_CODE = PATH_OUTPUT + "code/"
TRAIN_PATHS = []
TEST_PATHS = []


###########
# strings #
###########

DATA = 'data'
GROUND_TRUTH = 'ground_truth'
GROUND_WEIGHTS = 'ground_weights'
IS_TRAINING = 'is_training'
PRED = 'pred'
LOSS = 'loss'
TRAIN = 'train'
TEST = 'test'
