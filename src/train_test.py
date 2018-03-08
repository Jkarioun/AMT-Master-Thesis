from config import *
from utils import *
from data_utils import next_batch


def test(sess, model, placeholders, folder=PATH_VISUALISATION, rand_seed=10, create_images=True, onset=False,
         log_message=""):
    """ Run a test of the model.
    Reports the results in the log file and creates some images to facilitate results analysis.

    :param sess: session in which the model should run.
    :param model: cfr. model.get_model output.
    :param placeholders: cfr. model.get_model output.
    :param folder: directory in which to save the different image made (if create_images is True)
    :param rand_seed: seed for the random batch selection.
    :param create_images: True if images need to be created, else False.
    :param onset: True if the model is about predicting onsets, else False.
    :param log_message: String to add at the begining of each log line.
    """
    data_batch, ground_truth_batch_frame, ground_truth_batch_onset = next_batch(rand_seed, train=False)
    ground_truth_batch = ((ground_truth_batch_onset if onset else ground_truth_batch_frame) > 0).astype(int)

    ground_weights = onsets_and_frames_to_weights(ground_truth_batch_onset, ground_truth_batch_frame, onset=onset)

    prediction, loss_value = sess.run([model[PRED], model[LOSS]],
                                      feed_dict={placeholders[DATA]: data_batch,
                                                 placeholders[GROUND_TRUTH]: ground_truth_batch,
                                                 placeholders[GROUND_WEIGHTS]: ground_weights,
                                                 placeholders[IS_TRAINING]: False})

    if create_images:
        if not os.path.exists(folder):
            os.makedirs(folder)
        do_image(prediction[0], "01_Prediction", folder)
        do_image(data_batch[0][:, :, 0] * 3, "02_Input", folder)
        do_image(((ground_truth_batch_onset[0] > 0).astype(int) + (ground_truth_batch_frame[0] > 0)) / 2,
                 "03_Ground_truth_onsets_and_frames", folder)
        do_image((ground_truth_batch[0] - prediction[0] + 1) / 2, "04_Ground_Truth_vs_prediction", folder)
        do_image(prediction[0] > 0.5, "05_Prediction_treshold", folder)

        compare_tresh = np.empty((prediction.shape[1], prediction.shape[2] * 2))
        place = np.linspace(0, prediction.shape[2] * 2 - 2, prediction.shape[2], dtype=int)
        compare_tresh[:, place] = -ground_truth_batch[0]
        compare_tresh[:, place + 1] = prediction[0] > 0.5
        do_image((compare_tresh + 1) / 2, "06_compare_tresh", folder)

    cf_m = testing_metrics(ground_truth_batch[0] == 1, prediction[0] > 0.5)
    logging.info(log_message + "[rand_seed=%d][mode=testing][TP=%d][FP=%d][FN=%d][TN=%d][log_loss=%f]" % (
        rand_seed, cf_m['TP'], cf_m['FP'], cf_m['FN'], cf_m['TN'], loss_value))


def train(model, placeholders, num_batches=100, rand_seed=0, onset=False):
    """ Trains the model for num_batches batches and saves it in a file.
    Launch some testing periodically to have intermediary results. Reports some results in the logs.

    :param model: cfr. model.get_model output.
    :param placeholders: cfr. model.get_model output.
    :param num_batches: number of batch for training.
    :param rand_seed: batches random seed. Uses every batches whose seeds are in [rand_seed, rand_seed+num_batches[.
    :param onset: True if the model is about predicting onsets, else False.
    """
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(PATH_TENSORBOARD, sess.graph)
        if TRAIN_FROM_LAST:
            saver.restore(sess, PATH_CHECKPOINTS + CONFIG_NAME + ".ckpt")
        else:
            sess.run(init_op)

        for i in range(num_batches):
            data_batch, ground_truth_batch_frame, ground_truth_batch_onset = next_batch(i + rand_seed, train=True)
            ground_truth_batch = ((ground_truth_batch_onset if onset else ground_truth_batch_frame) > 0).astype(int)
            ground_weights = onsets_and_frames_to_weights(ground_truth_batch_onset, ground_truth_batch_frame,
                                                          onset=onset)
            loss_value, _ = sess.run([model[LOSS], model[TRAIN]],
                                     feed_dict={placeholders[DATA]: data_batch,
                                                placeholders[GROUND_TRUTH]: ground_truth_batch,
                                                placeholders[GROUND_WEIGHTS]: ground_weights,
                                                placeholders[IS_TRAINING]: True})

            logging.info("[rand_seed=%d][iteration=%d][mode=train][log_loss=%f]" % (
                rand_seed + i, i, loss_value))

            if (i + 1) % 10 == 0:
                # Save
                saver.save(sess, PATH_CHECKPOINTS + CONFIG_NAME + ".ckpt")
            if (i+1) % 50 == 0:
                # test
                test(sess, model, placeholders, folder=PATH_VISUALISATION + str(rand_seed) + "_" + str(i) + "/",
                     onset=ONSET, create_images=((i+1) % 5000 == 0), log_message="[iteration=%d]" % i)
                test(sess, model, placeholders, folder=PATH_VISUALISATION + str(rand_seed) + "_" + str(i) + "/",
                     onset=ONSET, create_images=False, log_message="[iteration=%d]" % i, rand_seed=i)

        # Save
        saver.save(sess, PATH_CHECKPOINTS + CONFIG_NAME + ".ckpt")
        writer.close()
