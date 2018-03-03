from config import *
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


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
    with tf.name_scope("harmonic_layer"):
        # Add an OOV pitch for zero padding (used during reordering)
        padding = tf.constant([[0, 0], [1, 1], [0, 1], [0, 0]])
        padded = tf.pad(inputs, padding, 'CONSTANT')

        # Copying (up to CONV_SIZE time) and reordering the data to use a conventional convolutive kernel
        pitch_first = tf.transpose(padded, [2, 0, 1, 3])
        reordering_indices = [[harmonic_mapping(i, bins_per_octave, int(inputs.shape[2]))] for i in
                              range(inputs.shape[2] * CONV_SIZE)]
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


def conv_net_kelz_modified(inputs):
    """Builds the ConvNet from Kelz 2016."""
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_AVG', uniform=True)):
        net = harmonic_layer(inputs, num_outputs=32, scope='conv1_mod', bins_per_octave=BINS_PER_OCTAVE)

        net = harmonic_layer(net, num_outputs=32, scope='conv2_mod', normalizer_fn=slim.batch_norm,
                             bins_per_octave=BINS_PER_OCTAVE)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2_mod')
        net = slim.dropout(net, 0.25, scope='dropout2_mod')

        net = harmonic_layer(net, 64, scope='conv3_mod', bins_per_octave=BINS_PER_OCTAVE / 2)
        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3_mod')
        net = slim.dropout(net, 0.25, scope='dropout3_mod')

        # Flatten while preserving batch and time dimensions.
        dims = tf.shape(net)
        net = tf.reshape(net, (dims[0], dims[1],
                               net.shape[2].value * net.shape[3].value), 'flatten4_mod')

        net = slim.fully_connected(net, 512, scope='fc5_mod')
        net = slim.dropout(net, 0.5, scope='dropout5_mod')

        net = slim.fully_connected(net, 88, activation_fn=tf.nn.sigmoid, scope='fc6_mod')

        return net


def get_model(input_data, ground_truth, kelz=False, hparams=DEFAULT_HPARAMS, onset=False):
    if kelz:
        output = conv_net_kelz(input_data)
    else:
        output = conv_net_kelz_modified(input_data)

    # loss
    loss = log_loss(ground_truth, output, onset=onset)

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

    # objective
    train = optimizer.minimize(loss)

    return output, loss, train


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


# Taken from tensorflow log_loss implementation to tweak its definition
def log_loss(labels, predictions, weights=1.0, epsilon=1e-7, scope=None,
             loss_collection=ops.GraphKeys.LOSSES,
             reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS, onset=False):
    """Adds a Log Loss term to the training procedure.

    `weights` acts as a coefficient for the loss. If a scalar is provided, then
    the loss is simply scaled by the given value. If `weights` is a tensor of size
    [batch_size], then the total loss for each sample of the batch is rescaled
    by the corresponding element in the `weights` vector. If the shape of
    `weights` matches the shape of `predictions`, then the loss of each
    measurable element of `predictions` is scaled by the corresponding value of
    `weights`.

    Args:
      labels: The ground truth output tensor, same dimensions as 'predictions'.
      predictions: The predicted outputs.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `losses` dimension).
      epsilon: A small increment to add to avoid taking a log of zero.
      scope: The scope for the operations performed in computing the loss.
      loss_collection: collection to which the loss will be added.
      reduction: Type of reduction to apply to loss.

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
      shape as `labels`; otherwise, it is scalar.

    Raises:
      ValueError: If the shape of `predictions` doesn't match that of `labels` or
        if the shape of `weights` is invalid.  Also if `labels` or `predictions`
        is None.
    """
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with ops.name_scope(scope, "log_loss",
                        (predictions, labels, weights)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        fact_loss = 100 if onset else 10
        losses = -math_ops.multiply(
            labels,
            fact_loss * math_ops.log(predictions + epsilon)) - math_ops.multiply(
            (1 - labels), math_ops.log(1 - predictions + epsilon))
        return tf.losses.compute_weighted_loss(
            losses, weights, scope, loss_collection, reduction=reduction)
