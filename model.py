from config import *


def harmonic_layer(inputs):
    padding = tf.constant([[0, 1], [0, 0], [0, 0]])
    padded = tf.pad(inputs, padding, 'CONSTANT')
    reordering = tf.gather_nd(padded, indices=HARMONIC_MAPPING.astype(int), name='Reordering')
    output = slim.conv2d(reordering, num_outputs=32, kernel_size=[5, 3], stride=5, padding='VALID')
    return output


def get_model(input_data):
    output = harmonic_layer(input_data)
    return output
