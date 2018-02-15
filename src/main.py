from init import *
from model import *

if __name__ == '__main__':
    init()

    input_data = np.zeros((2, 7, TOTAL_BIN, 1))
    input_data[0, 3, 0, 0] = 1

    data = tf.placeholder(tf.float32, shape=[None, None, TOTAL_BIN, 1], name='input')
    model = get_model(data)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Init
        sess.run(init_op)

        # Train
        result = sess.run(model, feed_dict={data: input_data})
        print(result.shape)
        plt.imshow(result[0, :, :])
        plt.show()
        # Test

        # print(result)
        # print(result.shape)
