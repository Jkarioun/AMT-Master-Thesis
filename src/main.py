from init import *
from model import *
from data_utils import *


def test(sess, model, loss, model_mod, loss_mod, folder="default", rand_seed=0):
    data_batch, ground_truth_batch = next_batch(rand_seed, train=False, onset=False)

    result, train_loss, result_mod, train_loss_mod = sess.run([model, loss, model_mod, loss_mod],
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

    do_image(result[0].T, "Kelz")
    do_image(result_mod[0].T, "Mod")
    do_image(ground_truth_batch[0].T, "Ground_Truth")
    do_image(ground_truth_batch[0].T - result_mod[0].T, "Ground_Truth__Mod")
    do_image(ground_truth_batch[0].T - result[0].T, "Ground_Truth__Kelz")
    do_image(result_mod[0].T > 0.5, "Mod_treshold")
    do_image(result[0].T > 0.5, "Kelz_treshold")

    compare_tresh = np.empty((result.shape[2] * 2, result.shape[1]))
    place = np.linspace(0, result.shape[2] * 2 - 2, result.shape[2], dtype=int)
    compare_tresh[place, :] = -ground_truth_batch[0].T
    compare_tresh[place + 1, :] = result_mod[0].T > 0.5
    do_image(compare_tresh, "compare_tresh")

    logging.info("kelz: " + str(train_loss))
    logging.info("mod : " + str(train_loss_mod))


if __name__ == '__main__':
    init()

    data = tf.placeholder(tf.float32, shape=[None, None, TOTAL_BIN, 1], name='input')
    ground_truth = tf.placeholder(tf.float32, shape=[None, None, PIANO_PITCHES])

    model, train, loss = get_model(data, ground_truth, kelz=True)
    model_mod, train_mod, loss_mod = get_model(data, ground_truth, kelz=False)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Train
    if TRAINING:
        with tf.Session() as sess:
            # Init
            sess.run(init_op)

            if TRAIN_FROM_LAST:
                saver.restore(sess, super_path)
            # Train
            for i in range(0):
                data_batch, ground_truth_batch = next_batch(i + RANDOM_DEBUG, train=True, onset=False)
                result, train_loss, result_mod, train_loss_mod = sess.run([train, loss, train_mod, loss_mod],
                                                                          feed_dict={data: data_batch,
                                                                                     ground_truth: ground_truth_batch})
                logging.info("Iteration %d" % (i + 1))
                logging.info("kelz: " + str(train_loss))
                logging.info("mod:  " + str(train_loss_mod))
                logging.info("ratio: " + str(train_loss_mod / train_loss))

                if i % 10 == 9:
                    # Save
                    saver.save(sess, super_path)
                if i % 100 == 99:
                    test(sess, model, loss, model_mod, loss_mod, folder=str(i)+"_"+str(RANDOM_DEBUG))

            # Save
            saver.save(sess, super_path)

    # Test
    with tf.Session() as sess:
        saver.restore(sess, super_path)
        test(sess, model, loss, model_mod, loss_mod)
