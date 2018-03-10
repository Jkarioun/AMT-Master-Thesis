from config import *
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def harmonic_layer(inputs, num_outputs, normalizer_fn=None, scope=None, bins_per_octave=BINS_PER_OCTAVE):
    """ Creates the harmonic convolution layer.

    :param inputs: input of the layer of the form [batch, frame, pitch, filters].
    :param num_outputs: number of output filters.
    :param normalizer_fn: cfr. slim.conv2d
    :param scope: cfr. slim.conv2d TODO : scope should take the entire layer
    :param bins_per_octave: number of bins per octave.
    :return: A tensorflow tensor of the form [batch, frame, pitch, num_outputs]
    """

    def harmonic_mapping(pos, out_of_range_pitch):
        """ Computes the pitch neuron that should be placed at position pos.

        :param pos: position for which we want a mapping.
        :param out_of_range_pitch: number of pitch neurons, which should be equal to the pitch number
                                   of the extra null-neuron.
        :return: The pitch neuron number to put at position pos.
        """
        base_position = pos // CONV_SIZE
        relative_position = round(math.log(HARMONIC_RELATIVES[pos % CONV_SIZE], 2) * bins_per_octave)
        ret = base_position + relative_position
        return ret if 0 <= ret < out_of_range_pitch else out_of_range_pitch

    with tf.name_scope("harmonic_layer"):
        # Add an OOV pitch for zero padding (used during reordering)
        padding = tf.constant([[0, 0], [1, 1], [0, 1], [0, 0]])
        padded = tf.pad(inputs, padding, 'CONSTANT')

        # Copying (up to CONV_SIZE time) and reordering the data to use a conventional convolutive kernel
        pitch_first = tf.transpose(padded, [2, 0, 1, 3])
        reordering_indices = [[harmonic_mapping(i, int(inputs.shape[2]))] for i in range(inputs.shape[2] * CONV_SIZE)]
        reordered_pitch_first = tf.gather_nd(pitch_first, indices=reordering_indices, name='Reordering')
        reordered = tf.transpose(reordered_pitch_first, [1, 2, 0, 3])

        output = slim.conv2d(reordered, num_outputs=num_outputs, kernel_size=[3, CONV_SIZE], stride=[1, CONV_SIZE],
                             padding='VALID', scope=scope, normalizer_fn=normalizer_fn)
    return output


def conv_net_kelz(inputs):
    """Builds the ConvNet from Kelz 2016, mainly copied from Magenta."""
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_AVG', uniform=True)), tf.name_scope("Kelz"):
        net = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
        net = slim.conv2d(
            net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
        net = slim.dropout(net, 0.25, scope='dropout2')

        net = slim.conv2d(net, 64, [3, 3], scope='conv3')
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


def conv_net_kelz_modified(placeholders):
    """Builds the ConvNet from Kelz 2016 with introduction of the harmonic_layer instead of the conventional convolution
    layer.

    :param placeholders: [DATA]: input of the net of the form [batch, frame, pitch, 1].
                         [IS_TRAINING]: True if training, else False.
    :return: Tensorflow tensor of the form [batch, frame, pitch] that corresponds to the prediction.
    """
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_AVG', uniform=True)):
        if FIRST_LAYER_HARMONIC:
            net = harmonic_layer(placeholders[DATA], num_outputs=32, scope='conv1_mod', bins_per_octave=BINS_PER_OCTAVE)
        else:
            net = slim.conv2d(placeholders[DATA], 32, [3, 3], scope='conv1_mod')

        net = harmonic_layer(net, num_outputs=32, scope='conv2_mod', normalizer_fn=slim.batch_norm,
                             bins_per_octave=BINS_PER_OCTAVE)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2_mod')
        net = slim.dropout(net, 0.25, scope='dropout2_mod', is_training=placeholders[IS_TRAINING])

        net = harmonic_layer(net, 64, scope='conv3_mod', bins_per_octave=BINS_PER_OCTAVE / 2)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3_mod')
        net = slim.dropout(net, 0.25, scope='dropout3_mod', is_training=placeholders[IS_TRAINING])

        # Flatten while preserving batch and time dimensions.
        dims = tf.shape(net)
        net = tf.reshape(net, (dims[0], dims[1],
                               net.shape[2].value * net.shape[3].value), 'flatten4_mod')

        net = slim.fully_connected(net, 512, scope='fc5_mod')
        net = slim.dropout(net, 0.5, scope='dropout5_mod', is_training=placeholders[IS_TRAINING])

        net = slim.fully_connected(net, 88, activation_fn=tf.nn.sigmoid, scope='fc6_mod')

        return net


def get_model(hparams=DEFAULT_HPARAMS):
    """ Creates the whole network.

    :param hparams: hparams.learning_rate should be the learning rate of the model.
    :return: placeholders: [DATA]: input of the net of the form [batch, frame, pitch, 1].
                           [GROUND_TRUTH]: ground_truth to be predicted of the form [batch, frame, pitch].
                           [GROUND_WEIGHTS]: weights to use to weight the loss, with same dimensions as [GROUND_TRUTH].
                           [IS_TRAINING]: boolean to say if the model is training or predicting.
             model: dictionary of tensorflow tensors corresponding to:
                    [PRED]: the prediction made by the model [batch, frame, pitch]
                    [LOSS]: the loss of the prediction (scalar)
                    [TRAIN]: the training neuron of the model (no return value).
    """
    placeholders = {DATA: tf.placeholder(tf.float32, shape=[None, None, TOTAL_BIN, 1], name='input'),
                    GROUND_TRUTH: tf.placeholder(tf.float32, shape=[None, None, PIANO_PITCHES]),
                    GROUND_WEIGHTS: tf.placeholder(tf.float32, shape=[None, None, PIANO_PITCHES]),
                    IS_TRAINING: tf.placeholder(tf.bool, shape=())}

    model = {}

    if KELZ_MODEL:
        model[PRED] = conv_net_kelz(placeholders[DATA])
    else:
        model[PRED] = conv_net_kelz_modified(placeholders)

    # loss
    model[LOSS] = tf.losses.log_loss(placeholders[GROUND_TRUTH], model[PRED], weights=placeholders[GROUND_WEIGHTS])

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    model[TRAIN] = optimizer.minimize(model[LOSS])

    return placeholders, model



#experimental code.
def get_phase_train_model(input_data, ground_truth, hparams=DEFAULT_HPARAMS):
    losses = []
    outputs = []
    trains = []

    def train_part(inputs, bins_per_pitch):
        print("---01---" + str(inputs.shape))
        inputs = inputs[:, :, :PIANO_PITCHES * BINS_PER_PITCH, :]
        print("---02---" + str(inputs.shape))
        dims = tf.shape(inputs)
        print("---1---" + str(dims))
        # TODO : verify the reshape keeps together the same pitch bins.
        net = tf.reshape(inputs, (dims[0], dims[1], PIANO_PITCHES, inputs.shape[3] * bins_per_pitch))
        net = slim.dropout(net, 0.5)
        print("---2---" + str(net.shape))
        net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
        print("---3---" + str(net.shape))
        outputs.append(tf.squeeze(net, [3]))
        print("---4---" + str(outputs[-1].shape))
        losses.append(tf.losses.log_loss(ground_truth, outputs[-1]))
        print("---5---" + str(losses[-1].shape))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        trains.append(optimizer.minimize(losses[-1]))

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_AVG', uniform=True)):
        with tf.name_scope('part_1'):
            conv_1 = harmonic_layer(input_data, num_outputs=32, bins_per_octave=BINS_PER_OCTAVE)
            with tf.name_scope('train_part_1'):
                train_part(conv_1, bins_per_pitch=BINS_PER_OCTAVE // 12)

        # with tf.name_scope('part_2'):
        #     conv_2 = harmonic_layer(conv_1, num_outputs=32, normalizer_fn=slim.batch_norm,
        #                             bins_per_octave=BINS_PER_OCTAVE)
        #     pool_2 = slim.max_pool2d(conv_2, [1, 2], stride=[1, 2])
        #     with tf.name_scope('train_part_2'):
        #         train_part(pool_2, bins_per_pitch=BINS_PER_OCTAVE // 24)
        #
        # with tf.name_scope('part_3'):
        #     drop_3 = slim.dropout(pool_2, 0.25)
        #     conv_3 = harmonic_layer(drop_3, 64, bins_per_octave=BINS_PER_OCTAVE / 2)
        #     pool_3 = slim.max_pool2d(conv_3, [1, 2], stride=[1, 2])
        #     with tf.name_scope('train_part_3'):
        #         train_part(pool_3, bins_per_pitch=BINS_PER_OCTAVE // 48)
        #
        # with tf.name_scope('part_4'):
        #     drop_41 = slim.dropout(pool_3, 0.25)
        #     # Flatten while preserving batch and time dimensions.
        #     dims = tf.shape(drop_41)
        #     shape_4 = tf.reshape(drop_41, (dims[0], dims[1],
        #                                    drop_41.shape[2].value * drop_41.shape[3].value), 'flatten4_mod')
        #     with tf.variable_scope('train_part_4'):
        #         outputs.append(slim.fully_connected(shape_4, 88))
        #         losses.append(tf.losses.log_loss(ground_truth, outputs[-1]))
        #         optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
        #         trains.append(optimizer.minimize(losses[-1]))

        return outputs, losses, trains
