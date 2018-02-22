from init import *
from model import *
from data_utils import *


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


                if i % 10 == 0:
                    # Save
                    saver.save(sess, super_path)

            # Save
            saver.save(sess, super_path)

    # Test
    with tf.Session() as sess:
        saver.restore(sess, super_path)

        i = 5
        data_batch, ground_truth_batch = next_batch(i, train=False, onset=False)

        result, train_loss, result_mod, train_loss_mod = sess.run([model, loss, model_mod, loss_mod],
                                                                  feed_dict={data: data_batch,
                                                                             ground_truth: ground_truth_batch})
        plt.imshow(result[0].T, aspect='auto')
        plt.title("Kelz")
        plt.show()
        plt.imshow(result_mod[0].T, aspect='auto')
        plt.title("Mod")
        plt.show()
        logging.info("kelz: " + str(train_loss))
        logging.info("mod: " + str(train_loss_mod))

