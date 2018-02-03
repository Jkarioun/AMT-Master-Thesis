import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

NUM_PITCHES = 88+6
BINS_PER_PITCH = 3
BINS_PER_OCTAVE = BINS_PER_PITCH*12
MIN_FREQ = librosa.note_to_hz('A0')*(2**(-1/BINS_PER_OCTAVE))
SR = 16000

CQT_WINDOW = 'hann'
CQT_HOP_LENGTH = 512

FRAME_PER_SEC = SR/CQT_HOP_LENGTH

def wav_to_CQT(filename):
	y, sr = librosa.core.load(filename, sr=SR)
	print("\n\n\n===========")
	C = abs(librosa.core.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, fmin=MIN_FREQ, n_bins=NUM_PITCHES*BINS_PER_PITCH, bins_per_octave=BINS_PER_OCTAVE, filter_scale=1, norm=1, sparsity=0, window=CQT_WINDOW, scale=True, pad_mode='constant'))
	print("===========\n\n\n")
	return C,sr

C,sr = wav_to_CQT("test.wav")
librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.show()