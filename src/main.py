from init import *
from model import *
from data_utils import *
from utils import do_image
from train_test import *

if __name__ == '__main__':
    init()

    data = tf.placeholder(tf.float32, shape=[None, None, TOTAL_BIN, 1], name='input')
    ground_truth = tf.placeholder(tf.float32, shape=[None, None, PIANO_PITCHES])
    placeholders = {'data': data, 'ground_truth': ground_truth}

    kelz_model, kelz_loss, kelz_train = get_model(data, ground_truth, kelz=True)
    # our_models, our_losses, our_trains = get_phase_train_model(data, ground_truth)
    our_model, our_loss, our_train = get_model(data, ground_truth, kelz=False)

    if TRAINING:
        train(kelz_model, kelz_loss, kelz_train, our_model, our_loss, our_train, placeholders, num_batches=NUM_BATCHES,
              rand_seed=RANDOM_DEBUG)

    # Test
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, super_path)
        for i in range(100):
            test(sess, kelz_model, kelz_loss, our_model, our_loss, placeholders, rand_seed=i, create_images=False)
