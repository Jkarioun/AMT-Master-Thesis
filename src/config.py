#!/usr/bin/python3.5

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

DISPLAY = True
if DISPLAY:
    import librosa.display
    import matplotlib.pyplot as plt

#
######################
# running parameters #
######################

CONFIG_NAME = 'kelz_batch_norm_test'
TRAINING = True
TESTING = False
TRAIN_FROM_LAST = False
show_images = True
RANDOM_DEBUG = 0
RAND_SEED = 10
NUM_BATCHES = 50000
MIN_FRAME_PER_BATCH = 1000
MAX_FRAME_PER_BATCH = 2000
CREATE_OUTPUT_FOLDERS = True

#
####################
# model parameters #
####################

ONSET = False
KELZ_MODEL = True
KELZ_KERNEL = [3, 3]
FIRST_LAYER_HARMONIC = False
HARMONIC_RELATIVES = np.array([1 / 3, 1 / 2, 1, 2, 3])
CONV_SIZE = len(HARMONIC_RELATIVES)
LEARNING_RATE = 0.0002
DEFAULT_HPARAMS = tf.contrib.training.HParams(learning_rate=LEARNING_RATE)

#
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

#
############
# data CQT #
############

# If want to center the BINS_PER_PITCH first bins around A0
# MIN_FREQ = librosa.note_to_hz('A0') * (2 ** (-1 / 2 + 1 / (2 * BINS_PER_PITCH)) / 12)
MIN_FREQ = librosa.note_to_hz('A0') * (2 ** (-1 / BINS_PER_OCTAVE))
SAMPLE_RESOLUTION = 16000
CQT_WINDOW = 'hann'
CQT_HOP_LENGTH = 512
FRAME_PER_SEC = SAMPLE_RESOLUTION / CQT_HOP_LENGTH

#
#########################
# data other parameters #
#########################

USE_ENSTDk = False
MIN_NOTE_LENGTH_IF_SUSTAIN = FRAME_PER_SEC * 2
MAX_NOTE_LENGTH = FRAME_PER_SEC * 5

#
#########
# Paths #
#########

PATH_OUTPUT = "../outputs/" + CONFIG_NAME + "/"
PATH_DEBUG = "../data/debug/"
PATH_MAPS = "../data/MAPS/"
PATH_SRC = '../src'
PATH_MAPS_PREPROCESSED = "../data/MAPS_PREPROCESSED/"
PATH_VISUALISATION = PATH_OUTPUT + "visualisation/"
PATH_TENSORBOARD = PATH_OUTPUT + "tensorboard/"
PATH_LOGS = PATH_OUTPUT + "logs/"
PATH_CHECKPOINTS = PATH_OUTPUT + "ckpt/"
PATH_CODE = PATH_OUTPUT + "code/"
TRAIN_FILENAMES = []
TEST_FILENAMES = []

#
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
