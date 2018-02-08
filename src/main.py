from init import *
from model import *

if __name__ == '__main__':
    init()

    input_data = np.ones((42, TOTAL_BIN, 20, 5))

    data = tf.placeholder(tf.float32, shape=input_data.shape, name='input')
    model = get_model(data)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        result = sess.run(model, feed_dict={data: input_data})
        print(result)
        print(result.shape)
