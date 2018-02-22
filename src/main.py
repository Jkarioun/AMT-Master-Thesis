from init import *
from model import *
from data_utils import *


def test(sess, kelz_model, kelz_loss, our_model, our_loss, folder="default", rand_seed=5):
    data_batch, ground_truth_batch = next_batch(rand_seed, train=False, onset=False)

    kelz_pred, kelz_loss_value, our_pred, our_loss_value = sess.run([kelz_model, kelz_loss, our_model, our_loss],
                                                                    feed_dict={data: data_batch,
                                                                               ground_truth: ground_truth_batch})

    if not os.path.exists(PATH_VISUALISATION + folder):
        os.makedirs(PATH_VISUALISATION + folder)

    def do_image(data, title):
        plt.imshow(data, aspect='auto')
        plt.title(title)
        plt.savefig(PATH_VISUALISATION + folder + '/' + title + '.png')
        if show_images:
            plt.show()

    do_image(kelz_pred[0].T, "Kelz")
    do_image(our_pred[0].T, "Mod")
    do_image(data_batch[0][:, :, 0].T, "Input")
    do_image(ground_truth_batch[0].T, "Ground_Truth")
    do_image(ground_truth_batch[0].T - our_pred[0].T, "Ground_Truth__Mod")
    do_image(ground_truth_batch[0].T - kelz_pred[0].T, "Ground_Truth__Kelz")
    do_image(our_pred[0].T > 0.5, "Mod_treshold")
    do_image(kelz_pred[0].T > 0.5, "Kelz_treshold")

    compare_tresh = np.empty((kelz_pred.shape[2] * 2, kelz_pred.shape[1]))
    place = np.linspace(0, kelz_pred.shape[2] * 2 - 2, kelz_pred.shape[2], dtype=int)
    compare_tresh[place, :] = -ground_truth_batch[0].T
    compare_tresh[place + 1, :] = our_pred[0].T > 0.5
    do_image(compare_tresh, "compare_tresh")

    logging.info("kelz: " + str(kelz_loss_value))
    logging.info("mod : " + str(our_loss_value))


def train(kelz_model, kelz_loss, kelz_train, our_model, our_loss, our_train, num_batches=100, rand_seed=0, onset=False):
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("\\tmp\\output", sess.graph)
        if TRAIN_FROM_LAST:
            saver.restore(sess, super_path)
        else:
            sess.run(init_op)

        for i in range(num_batches):
            data_batch, ground_truth_batch = next_batch(i + rand_seed, train=True, onset=onset)
            kelz_loss_value, _, our_loss_value, _ = sess.run([kelz_loss, kelz_train, our_loss, our_train],
                                                             feed_dict={data: data_batch,
                                                                        ground_truth: ground_truth_batch})
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
            if i % 10 == 9:
                test(sess, kelz_model, kelz_loss, our_model, our_loss, folder=str(i) + "_" + str(rand_seed))

        # Save
        saver.save(sess, super_path)
        writer.close()


if __name__ == '__main__':
    init()

    data = tf.placeholder(tf.float32, shape=[None, None, TOTAL_BIN, 1], name='input')
    ground_truth = tf.placeholder(tf.float32, shape=[None, None, PIANO_PITCHES])

    kelz_model, kelz_loss, kelz_train = get_model(data, ground_truth, kelz=True)
    our_models, our_losses, our_trains = get_phase_train_model(data, ground_truth)

    if TRAINING:
        train(kelz_model, kelz_loss, kelz_train, our_models[0], our_losses[0], our_trains[0], num_batches=NUM_BATCHES,
              rand_seed=RANDOM_DEBUG)

    # Test
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, super_path)
        test(sess, kelz_model, kelz_loss, our_models[0], our_losses[0])
