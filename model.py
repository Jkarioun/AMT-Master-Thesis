from config import *


def harmonic_layer(inputs,
                   num_outputs,
                   normalizer_fn=None,
                   scope=None):
    padding = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]])
    padded = tf.pad(inputs, padding, 'CONSTANT')

    reordering_indices = [[[i, int(j)] for j in HARMONIC_MAPPING] for i in range(inputs.shape[0])]

    reordered = tf.gather_nd(padded, indices=reordering_indices, name='Reordering')

    output = slim.conv2d(reordered, num_outputs=num_outputs, kernel_size=[5, 3], stride=[5, 1], padding='VALID',
                         scope=scope,
                         normalizer_fn=normalizer_fn)
    print(output)
    return output


# Copied from Magenta
def conv_net_kelz(inputs):
    """Builds the ConvNet from Kelz 2016."""
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_AVG', uniform=True)):
        net = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
        print(net)
        net = slim.conv2d(
            net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm)
        print(net)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
        net = slim.dropout(net, 0.25, scope='dropout2')

        net = slim.conv2d(net, 64, [3, 3], scope='conv3')
        print(net)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3')
        net = slim.dropout(net, 0.25, scope='dropout3')

        # Flatten while preserving batch and time dimensions.
        dims = tf.shape(net)
        net = tf.reshape(net, (dims[0], dims[1],
                               net.shape[2].value * net.shape[3].value), 'flatten4')

        net = slim.fully_connected(net, 512, scope='fc5')
        net = slim.dropout(net, 0.5, scope='dropout5')

        return net


def conv_net_kelz_modified(inputs):
    """Builds the ConvNet from Kelz 2016."""
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_AVG', uniform=True)):
        net = harmonic_layer(inputs, num_outputs=32, scope='conv1')

        net = harmonic_layer(net, num_outputs=32, scope='conv2', normalizer_fn=slim.batch_norm)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
        net = slim.dropout(net, 0.25, scope='dropout2')

        net = harmonic_layer(net, 64, scope='conv3')
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3')
        net = slim.dropout(net, 0.25, scope='dropout3')

        # Flatten while preserving batch and time dimensions.
        dims = tf.shape(net)
        net = tf.reshape(net, (dims[0], dims[1],
                               net.shape[2].value * net.shape[3].value), 'flatten4')

        net = slim.fully_connected(net, 512, scope='fc5')
        net = slim.dropout(net, 0.5, scope='dropout5')

        return net


def get_model(input_data):
    output = conv_net_kelz(input_data)
    return output
