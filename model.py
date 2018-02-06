from config import *


def harmonic_layer(inputs):
    padding = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]])
    padded = tf.pad(inputs, padding, 'CONSTANT')

    reordering_tuples = np.zeros((inputs.shape[0], HARMONIC_MAPPING.size, 2))
    for i in range(inputs.shape[0]):
        for j in range(HARMONIC_MAPPING.size):
            reordering_tuples[i][j] = [i, HARMONIC_MAPPING[j]]

    reordering = tf.gather_nd(padded, indices=reordering_tuples.astype(int), name='Reordering')
    output = slim.conv2d(reordering, num_outputs=32, kernel_size=[5, 3], stride=5, padding='VALID')
    return output


def get_model(input_data):
    output = harmonic_layer(input_data)
    return output
