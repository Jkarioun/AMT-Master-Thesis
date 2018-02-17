from config import *


def harmonic_mapping(i, bins_per_octave, OOV_pitch):
    base_position = i // CONV_SIZE
    relative_position = round(math.log(HARMONIC_RELATIVES[i % CONV_SIZE], 2) * bins_per_octave)
    ret = base_position + relative_position
    return ret if 0 <= ret < OOV_pitch else OOV_pitch


def harmonic_layer(inputs,
                   num_outputs,
                   normalizer_fn=None,
                   scope=None,
                   bins_per_octave=BINS_PER_OCTAVE):
    # Add an OOV pitch for zero padding (used during reordering)
    padding = tf.constant([[0, 0], [1, 1], [0, 1], [0, 0]])
    padded = tf.pad(inputs, padding, 'CONSTANT')

    # Copying (up to CONV_SIZE time) and reordering the data to use a conventional convolutive kernel
    pitch_first = tf.transpose(padded, [2, 0, 1, 3])
    reordering_indices = [[harmonic_mapping(i, bins_per_octave, int(inputs.shape[2]))] for i in range(inputs.shape[2] * CONV_SIZE)]
    reordered_pitch_first = tf.gather_nd(pitch_first, indices=reordering_indices, name='Reordering')
    reordered = tf.transpose(reordered_pitch_first, [1, 2, 0, 3])

    output = slim.conv2d(reordered, num_outputs=num_outputs, kernel_size=[3, CONV_SIZE], stride=[1, CONV_SIZE],
                         padding='VALID',
                         scope=scope,
                         normalizer_fn=normalizer_fn)
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

        net = slim.fully_connected(net, 88, activation_fn=tf.nn.sigmoid, scope='fc6')

        return net


def conv_net_kelz_modified(inputs):
    """Builds the ConvNet from Kelz 2016."""
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_AVG', uniform=True)):
        net = harmonic_layer(inputs, num_outputs=32, scope='conv1', bins_per_octave=BINS_PER_OCTAVE)

        net = harmonic_layer(net, num_outputs=32, scope='conv2', normalizer_fn=slim.batch_norm,
                             bins_per_octave=BINS_PER_OCTAVE)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
        net = slim.dropout(net, 0.25, scope='dropout2')

        net = harmonic_layer(net, 64, scope='conv3', bins_per_octave=BINS_PER_OCTAVE / 2)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3')
        net = slim.dropout(net, 0.25, scope='dropout3')

        # Flatten while preserving batch and time dimensions.
        dims = tf.shape(net)
        net = tf.reshape(net, (dims[0], dims[1],
                               net.shape[2].value * net.shape[3].value), 'flatten4')

        net = slim.fully_connected(net, 512, scope='fc5')
        net = slim.dropout(net, 0.5, scope='dropout5')

        net = slim.fully_connected(net, 88, activation_fn=tf.nn.sigmoid, scope='fc6')

        return net


def get_model(input_data, ground_truth, hparams=DEFAULT_HPARAMS):
    # output = conv_net_kelz(input_data)
    # batch x time x 88
    output = conv_net_kelz_modified(input_data)

    # loss
    loss = tf.reduce_mean(tf.square(ground_truth - output))
    loss = tf.losses.log_loss(ground_truth, output)

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

    # objective
    train = optimizer.minimize(loss)

    return output, train, loss
