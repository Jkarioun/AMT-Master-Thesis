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
    example_ground_truth = [midi_file_to_tensor(PATH_DEBUG + "example.mid")[:]]
    print(example_data.shape)
    print(example_ground_truth[0].shape)
    plt.imshow(example_data[0,:,:,0])
    plt.show()
    plt.imshow(example_ground_truth[0,:,:])
    plt.show()
    input()

    model, train = get_model(data, ground_truth)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Train
    if True:
        with tf.Session() as sess:
            # Init
            sess.run(init_op)

            # Train
            for _ in range(1):
                result = sess.run(train, feed_dict={data: example_data, ground_truth: example_ground_truth})

            print(result.shape)
            plt.imshow(result[0, :, :])
            plt.show()
            # Test

            # print(result)
            # print(result.shape)

            # Save
            saver.save(sess, "./tmp/model.ckpt")

    # Test
    if False:
        with tf.Session() as sess:
            saver.restore(sess, "./tmp/model.ckpt")
