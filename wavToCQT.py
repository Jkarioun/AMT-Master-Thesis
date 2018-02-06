import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

NUM_PITCHES = 88+24
BINS_PER_PITCH = 3
BINS_PER_OCTAVE = BINS_PER_PITCH*12
MIN_FREQ = librosa.note_to_hz('A0')*(2**(-1/BINS_PER_OCTAVE))
SR = 44100

CQT_WINDOW = 'hann'
CQT_HOP_LENGTH = 512

FRAME_PER_SEC = SR/CQT_HOP_LENGTH

def wav_to_CQT(filename):
	y, sr = librosa.core.load(filename, sr=SR)
	print("\n\n\n===========")
	C = abs(librosa.core.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, fmin=MIN_FREQ, n_bins=NUM_PITCHES*BINS_PER_PITCH, bins_per_octave=BINS_PER_OCTAVE, filter_scale=1, norm=1, sparsity=0, window=CQT_WINDOW, scale=True, pad_mode='constant'))
	print("===========\n\n\n")
	return C,sr

# Copied from magenta : https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
def conv_net_kelz(inputs):
  """Builds the ConvNet from Kelz 2016."""
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_AVG', uniform=True)):
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

    return net


C,sr = wav_to_CQT("test.wav")
librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.show()