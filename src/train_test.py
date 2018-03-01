from config import *
from utils import do_image
from data_utils import next_batch


def test(sess, kelz_model, kelz_loss, our_model, our_loss, placeholders, folder="default", rand_seed=10, create_images=True, onset=False):
    data_batch, ground_truth_batch_frame, ground_truth_batch_onset = next_batch(rand_seed, train=False)
    ground_truth_batch = ((ground_truth_batch_onset if onset else ground_truth_batch_frame) > 0).astype(int)

    kelz_pred, kelz_loss_value, our_pred, our_loss_value = sess.run([kelz_model, kelz_loss, our_model, our_loss],
                                                                    feed_dict={placeholders['data']: data_batch,
                                                                               placeholders['ground_truth']: ground_truth_batch})

    if not os.path.exists(PATH_VISUALISATION + folder):
        os.makedirs(PATH_VISUALISATION + folder)

    if create_images:
        do_image(kelz_pred[0].T, "01Kelz", folder)
        do_image(ground_truth_batch_frame[0].T > 0, "02Ground_Truth_Frame", folder)
        do_image(our_pred[0].T, "03Mod", folder)
        do_image(data_batch[0][:, :, 0].T, "04Input", folder)
        do_image(ground_truth_batch_onset[0].T > 0, "05Ground_Truth_Onset", folder)
        do_image(((ground_truth_batch_onset[0].T > 0).astype(int)+(ground_truth_batch_frame[0].T > 0))/2, "05Ground_Truth_Onset_And_frame", folder)
        do_image((ground_truth_batch[0].T - our_pred[0].T + 1)/2, "06Ground_Truth__Mod", folder)
        do_image((ground_truth_batch[0].T - kelz_pred[0].T + 1)/2, "07Ground_Truth__Kelz", folder)
        do_image(our_pred[0].T > 0.5, "08Mod_treshold", folder)
        do_image(kelz_pred[0].T > 0.5, "09Kelz_treshold", folder)

        compare_tresh = np.empty((kelz_pred.shape[2] * 2, kelz_pred.shape[1]))
        place = np.linspace(0, kelz_pred.shape[2] * 2 - 2, kelz_pred.shape[2], dtype=int)
        compare_tresh[place, :] = -ground_truth_batch[0].T
        compare_tresh[place + 1, :] = our_pred[0].T > 0.5
        do_image((compare_tresh+1)/2, "10compare_tresh", folder)

    logging.info("kelz: " + str(kelz_loss_value))
    logging.info("mod : " + str(our_loss_value))


def train(kelz_model, kelz_loss, kelz_train, our_model, our_loss, our_train, placeholders, num_batches=100, rand_seed=0, onset=False):
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("..\\tmp\\output", sess.graph)
        if TRAIN_FROM_LAST:
            saver.restore(sess, super_path)
        else:
            sess.run(init_op)

        for i in range(num_batches):
            data_batch, ground_truth_batch_frame, ground_truth_batch_onset = next_batch(i + rand_seed, train=True)
            ground_truth_batch = ((ground_truth_batch_onset if onset else ground_truth_batch_frame) > 0).astype(int)
            kelz_loss_value, _, our_loss_value, _ = sess.run([kelz_loss, kelz_train, our_loss, our_train],
                                                             feed_dict={placeholders['data']: data_batch,
                                                                        placeholders['ground_truth']: ground_truth_batch})
            logging.info("Iteration %d" % (i + 1))
            logging.info("kelz: " + str(kelz_loss_value))
            logging.info("mod:  " + str(our_loss_value))
            logging.info("ratio: " + str(our_loss_value / kelz_loss_value))

            print("Iteration %d" % (i + 1))
            print("kelz: " + str(kelz_loss_value))
            print("mod:  " + str(our_loss_value))
            print("ratio: " + str(our_loss_value / kelz_loss_value))

            if i % 10 == 9:
                # Save
                saver.save(sess, super_path)
            if i % 500 == 0:
                test(sess, kelz_model, kelz_loss, our_model, our_loss, placeholders, folder=str(i) + "_" + str(rand_seed))

        # Save
        saver.save(sess, super_path)
        writer.close()