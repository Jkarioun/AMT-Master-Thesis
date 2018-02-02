import tensorflow as tf
import scipy
import scipy.io.wavfile as sio
import six


# START: taken from magenta: https://github.com/tensorflow/magenta/tree/master/magenta/models/onsets_frames_transcription

#MIN_MIDI_PITCH = librosa.note_to_midi('A0')
#MAX_MIDI_PITCH = librosa.note_to_midi('C8')
#MIDI_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1

DEFAULT_CQT_BINS_PER_OCTAVE = 36
DEFAULT_JITTER_AMOUNT_MS = 0
DEFAULT_JITTER_WAV_AND_LABEL_SEPARATELY = False
DEFAULT_MIN_FRAME_OCCUPANCY_FOR_LABEL = 0.0
DEFAULT_NORMALIZE_AUDIO = False
DEFAULT_ONSET_DELAY = 0
DEFAULT_ONSET_LENGTH = 100
DEFAULT_ONSET_MODE = 'window'
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SPEC_FMIN = 30.0
DEFAULT_SPEC_HOP_LENGTH = 512
DEFAULT_SPEC_LOG_AMPLITUDE = True
DEFAULT_SPEC_N_BINS = 229
DEFAULT_SPEC_TYPE = 'mel'


DEFAULT_HPARAMS = tf.contrib.training.HParams(
    cqt_bins_per_octave=DEFAULT_CQT_BINS_PER_OCTAVE,
    jitter_amount_ms=DEFAULT_JITTER_AMOUNT_MS,
    jitter_wav_and_label_separately=DEFAULT_JITTER_WAV_AND_LABEL_SEPARATELY,
    min_frame_occupancy_for_label=DEFAULT_MIN_FRAME_OCCUPANCY_FOR_LABEL,
    normalize_audio=DEFAULT_NORMALIZE_AUDIO,
    onset_delay=DEFAULT_ONSET_DELAY,
    onset_length=DEFAULT_ONSET_LENGTH,
    onset_mode=DEFAULT_ONSET_MODE,
    sample_rate=DEFAULT_SAMPLE_RATE,
    spec_fmin=DEFAULT_SPEC_FMIN,
    spec_hop_length=DEFAULT_SPEC_HOP_LENGTH,
    spec_log_amplitude=DEFAULT_SPEC_LOG_AMPLITUDE,
    spec_n_bins=DEFAULT_SPEC_N_BINS,
    spec_type=DEFAULT_SPEC_TYPE,
)

class AudioIOException(BaseException):
  pass


class AudioIOReadException(AudioIOException):
  pass


def wav_data_to_samples(wav_data, sample_rate):
  """Read PCM-formatted WAV data and return a NumPy array of samples.
  Uses scipy to read and librosa to process WAV data. Audio will be converted to
  mono if necessary.
  Args:
    wav_data: WAV audio data to read.
    sample_rate: The number of samples per second at which the audio will be
        returned. Resampling will be performed if necessary.
  Returns:
    A numpy array of audio samples, single-channel (mono) and sampled at the
    specified rate, in float32 format.
  Raises:
    AudioIOReadException: If scipy is unable to read the WAV data.
    AudioIOException: If audio processing fails.
  """
  try:
    # Read the wav file, converting sample rate & number of channels.
    native_sr, y = sio.read(six.BytesIO(wav_data))
  except Exception as e:  # pylint: disable=broad-except
    raise AudioIOReadException(e)
  if y.dtype != np.int16:
    raise AudioIOException('WAV file not 16-bit PCM, unsupported')
  try:
    # Convert to float, mono, and the desired sample rate.
    y = int16_samples_to_float32(y)
    if y.ndim == 2 and y.shape[1] == 2:
      y = y.T
      y = librosa.to_mono(y)
    if native_sr != sample_rate:
      y = librosa.resample(y, native_sr, sample_rate)
  except Exception as e:  # pylint: disable=broad-except
    raise AudioIOException(e)
  return y



def _wav_to_cqt(wav_audio, hparams):
  """Transforms the contents of a wav file into a series of CQT frames."""
  y = wav_data_to_samples(wav_audio, hparams.sample_rate)

  cqt = np.abs(
      librosa.core.cqt(
          y,
          hparams.sample_rate,
          hop_length=hparams.spec_hop_length,
          fmin=hparams.spec_fmin,
          n_bins=hparams.spec_n_bins,
          bins_per_octave=hparams.cqt_bins_per_octave,
          real=False),
      dtype=np.float32)

  # Transpose so that the data is in [frame, bins] format.
  cqt = cqt.T
  return cqt
# END: taken from magenta



print(_wav_to_cqt("test.wav", DEFAULT_HPARAMS))