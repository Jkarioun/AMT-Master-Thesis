#!/usr/bin/python3.5

from init import *
from model import *
from data_utils import *
from utils import do_image
from train_test import *

if __name__ == '__main__':
    init()

    placeholders, model = get_model(hparams=DEFAULT_HPARAMS)

    if TRAINING:
        train(model, placeholders, num_batches=NUM_BATCHES, rand_seed=RANDOM_DEBUG, onset=ONSET)

    if TESTING:
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, PATH_CHECKPOINTS + CONFIG_NAME + ".ckpt")
            for i in range(100):
                test(sess, model, placeholders, rand_seed=i, create_images=False, onset=ONSET,
                     folder=PATH_VISUALISATION + "testing/")
