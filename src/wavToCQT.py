from src.config import *


def wav_to_CQT(filename):
    y, sr = librosa.core.load(filename, sr=SAMPLE_RESOLUTION)
    print("\n\n\n===========")
    cqt = abs(librosa.core.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, fmin=MIN_FREQ, n_bins=NUM_PITCHES * BINS_PER_PITCH,
                               bins_per_octave=BINS_PER_OCTAVE, filter_scale=1, norm=1, sparsity=0, window=CQT_WINDOW,
                               scale=True, pad_mode='constant'))
    print("===========\n\n\n")
    return cqt, sr


C, sr = wav_to_CQT("test.wav")
librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.show()
