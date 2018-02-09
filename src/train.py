from config import *


def train(examples_path, hparams=DEFAULT_HPARAMS):
    with tf.Graph().as_default():
        transcription_data = _get_data(examples_path, hparams, is_training=True)

        loss, losses, unused_labels, unused_predictions, images = model.get_model(
            transcription_data, hparams, is_training=True)

        tf.summary.scalar('loss', loss)
        for label, loss_collection in losses.iteritems():
            loss_label = 'losses/' + label
            tf.summary.scalar(loss_label, tf.reduce_mean(loss_collection))
        for name, image in images.iteritems():
            tf.summary.image(name, image)
        optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

        train_op = slim.learning.create_train_op(
            loss,
            optimizer,
            clip_gradient_norm=hparams.clip_norm,
            summarize_gradients=True)

        return train_op
