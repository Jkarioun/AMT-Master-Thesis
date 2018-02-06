from config import *


def harmonic_layer(inputs):
    padding = tf.constant([[0, 1], [0, 0], [0, 0]])
    padded = tf.pad(inputs, padding, 'CONSTANT')
    print(padded)
    output = slim.conv2d(padded, num_outputs=32, kernel_size=[5, 3], stride=5, padding='VALID')
    return output


def get_model(input_data):
    output = harmonic_layer(input_data)
    return output
