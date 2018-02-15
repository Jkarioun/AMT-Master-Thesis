from init import *
from model import *
from data_utils import *

if __name__ == '__main__':
    init()

    data = tf.placeholder(tf.float32, shape=[None, None, TOTAL_BIN, 1], name='input')
    ground_truth = tf.placeholder(tf.float32, shape=[None, None, PIANO_PITCHES])

    # examples
    example_data, _ = wav_to_CQT(PATH_DEBUG + "example.wav")
    example_data = np.reshape(example_data, [1, -1, TOTAL_BIN, 1])
    example_ground_truth = np.zeros((1, example_data.shape[1]-6, example_data.shape[2]))
    intermediary_variable = midi_file_to_tensor(PATH_DEBUG + "example.mid")
    example_ground_truth[0, 3:intermediary_variable.shape[0]+3, :intermediary_variable.shape[1]] = intermediary_variable

    #example_ground_truth = np.zeros((1, example_data.shape[1], example_data.shape[2]))
    #intermediary_variable = midi_file_to_tensor(PATH_DEBUG + "example.mid")
    #example_ground_truth[0, :intermediary_variable.shape[0], :intermediary_variable.shape[1]] = intermediary_variable

    model, train, loss = get_model(data, ground_truth)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    TRAINING = True

    super_path = "./tmp/ours2.ckpt"

    # Train
    if TRAINING:
        with tf.Session() as sess:
            # Init
            sess.run(init_op)

            #saver.restore(sess, "./tmp/kelz.ckpt")
            # Train
            for i in range(1000):
                result, train_loss = sess.run([train, loss], feed_dict={data: example_data, ground_truth: example_ground_truth})
                print("Iteration %d" % i)
                print(train_loss)

            # Test

            # print(result)
            # print(result.shape)

            # Save
            saver.save(sess, super_path)

    # Test
    else:
        with tf.Session() as sess:
            saver.restore(sess, super_path)

            result = sess.run(model, feed_dict={data: example_data})
            np.set_printoptions(threshold=np.nan)
            print(result)
            plt.imshow(result[0].T)
            plt.show()

            plt.imshow(example_ground_truth[0].T)
            plt.show()
