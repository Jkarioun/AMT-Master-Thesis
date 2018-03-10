from config import *
from init import init
from utils import do_image


def wav_to_CQT(file):
    """ Computes the CQT of a signal.

    :param file: file-like object following the .wav format.
    :return: the CQT of the signal in a numpy array with the form [frame, pitch].
    """
    tmp = io.BytesIO(file.read())
    y, sr = sf.read(tmp)
    y = librosa.core.to_mono(y.T)
    y = librosa.core.resample(y, orig_sr=sr, target_sr=SAMPLE_RESOLUTION)
    sr = SAMPLE_RESOLUTION
    cqt = abs(librosa.core.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, fmin=MIN_FREQ, n_bins=NUM_PITCHES * BINS_PER_PITCH,
                               bins_per_octave=BINS_PER_OCTAVE, filter_scale=1, norm=1, sparsity=0, window=CQT_WINDOW,
                               scale=True, pad_mode='constant'))
    return cqt.T, sr


def display_CQT(cqt, sr):
    """ Displays the CQT of a signal.

    :param cqt: numpy array of a CQT with the form [frame, pitch].
    :param sr: sampling rate of the CQT.
    """
    if DISPLAY:
        librosa.display.specshow(librosa.amplitude_to_db(cqt.T, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrum')
        plt.show()
    else:
        print("DISPLAY is set to False")


def compare_midi(gold_MIDI_path, pred_MIDI_path, output_path):
    """ Creates an image that allows easy midi comparison.

    :param gold_MIDI_path: path to the first midi file to compare.
    :param pred_MIDI_path: path to the second midi file to compare.
    :param output_path: path of the resulting image.
    """
    PIX_PER_PITCH = 15
    MIN_PITCH = 21
    MAX_PITCH = 108
    PIX_PER_SEC = 100
    SPACE_PRED = 4
    PRED_COLOR = (0, 0, 255)
    GOLD_COLOR = (255, 0, 0)
    pitch_range = MAX_PITCH - MIN_PITCH + 1
    pitch_pixs = (pitch_range + 1) * PIX_PER_PITCH

    def pitch2pix(pitch):
        return pitch_pixs - ((pitch - MIN_PITCH + 1) * PIX_PER_PITCH)

    def time2pix(sec):
        return int(sec * PIX_PER_SEC)

    def set_output(outputs, note, pred):
        for i in range(time2pix(note.start), time2pix(note.end) + 1):
            outputs[pitch2pix(note.pitch) + (SPACE_PRED if pred else 0)][i] = PRED_COLOR if pred else GOLD_COLOR
            outputs[pitch2pix(note.pitch) - (SPACE_PRED if pred else 0)][i] = PRED_COLOR if pred else GOLD_COLOR
        for i in range(-SPACE_PRED, SPACE_PRED + 1):
            outputs[i + pitch2pix(note.pitch)][time2pix(note.start)] = PRED_COLOR if pred else GOLD_COLOR
            outputs[i + pitch2pix(note.pitch)][time2pix(note.end)] = PRED_COLOR if pred else GOLD_COLOR

    gold_MID = pm.PrettyMIDI(gold_MIDI_path)
    pred_MID = pm.PrettyMIDI(pred_MIDI_path)

    end = max(gold_MID.get_end_time(), pred_MID.get_end_time())
    length = int((end + 1) * PIX_PER_SEC)
    im = Image.new("RGB", (length, pitch_pixs))
    output = [[(255, 255, 255) for _ in range(length)] for _ in range(pitch_pixs)]

    # reference pitch-lines
    for i in range(length):
        for j in [43, 47, 50, 53, 57, 64, 67, 71, 74, 77]:
            output[pitch2pix(j)][i] = (0, 255, 0)

    # reference time-lines:
    for i in range(pitch_pixs):
        for j in range(0, int(length / PIX_PER_SEC)):
            output[i][time2pix(j)] = (240, 240, 240)

    for inst in pred_MID.instruments:
        for note in inst.notes:
            set_output(output, note, True)

    for inst in gold_MID.instruments:
        for note in inst.notes:
            set_output(output, note, False)

    im.putdata([item for sublist in output for item in sublist])
    im.save(output_path)


def midi_file_to_tensor(file):
    """Reads a midi file and returns its content as a tensor

    :param file: file-like object following the midi format. The file must contain only one instrument.
    :return: a tensor of the form output[onset, frame, pitch] =
        onset_velocity_during_this_frame if onset==1 else pitch_velocity_during_this_frame.
    """
    midi = pm.PrettyMIDI(io.BytesIO(file.read()))
    frames = math.floor(midi.get_end_time() * FRAME_PER_SEC) + 1
    output = np.full((2, frames, PIANO_PITCHES), fill_value=0, dtype=int)
    output[0, :-1, :] = midi.get_piano_roll(FRAME_PER_SEC)[PIANO_MIN_PITCH:PIANO_MAX_PITCH + 1, :].T
    for note in midi.instruments[0].notes:
        output[:, math.floor(note.start * FRAME_PER_SEC), note.pitch - PIANO_MIN_PITCH] = note.velocity
    return output


def next_batch(seed, train=True):
    """ Creates a batch from the MAPS dataset, concatenating sub_batches until expected batch size is obtained.
    init.init() must be called before use.

    :param seed: seed for the randomness of the batch creation.
    :param train: True if a training batch is required, False if a testing batch is required.
    :return: the input (1,#frame,TOTAL_BIN,1),
             expected framewise activation output (1,#frame,PIANO_PITCHES) and
             expected onset activation output (1,#frame,PIANO_PITCHES)
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)
    data_batch, ground_truth_batch_frame, ground_truth_batch_onset = util_next_batch(train=train)

    # appends sub_batches until there are enough frames
    while data_batch.shape[0] <= MIN_FRAME_PER_BATCH:
        inputs, outputs_frame, outputs_onset = util_next_batch(train=train)
        data_batch = np.concatenate((data_batch, inputs))
        ground_truth_batch_frame = np.concatenate((ground_truth_batch_frame, outputs_frame))
        ground_truth_batch_onset = np.concatenate((ground_truth_batch_onset, outputs_onset))

    # suppresses exceeding frames
    if data_batch.shape[0] > MAX_FRAME_PER_BATCH:
        data_batch = data_batch[:MAX_FRAME_PER_BATCH]
        ground_truth_batch_frame = ground_truth_batch_frame[:MAX_FRAME_PER_BATCH]
        ground_truth_batch_onset = ground_truth_batch_onset[:MAX_FRAME_PER_BATCH]

    return np.expand_dims(data_batch, axis=0), \
           np.expand_dims(ground_truth_batch_frame, axis=0), \
           np.expand_dims(ground_truth_batch_onset, axis=0)


def util_next_batch(train=True, music_name=None):
    """ Creates a sub_batch from a single MAPS music.

    :param train: True if a training batch is required, False if a testing batch is required.
    :param music_name: The music from which to do the sub_batch
                       If set to None, the music is chosen at random.
    :return: the input (#frame,TOTAL_BIN,1), framewise activation output (#frame,PIANO_PITCHES)
            and onset activation output (#frame,PIANO_PITCHES)
    """
    if music_name is None:
        if train:
            music_name = TRAIN_FILENAMES[np.random.randint(0, len(TRAIN_FILENAMES))]
        else:
            music_name = TEST_FILENAMES[np.random.randint(0, len(TEST_FILENAMES))]

    print(music_name)

    # input
    data_batch = np.load(PATH_MAPS_PREPROCESSED + music_name + '.npy')
    data_batch = np.reshape(data_batch, [-1, TOTAL_BIN, 1])

    # expected output
    with open(PATH_MAPS_PREPROCESSED + music_name + '.mid', 'rb') as mid_file:
        unpadded_tensor = midi_file_to_tensor(mid_file)
        if unpadded_tensor.shape[1] > MAX_FRAME_PER_BATCH:
            unpadded_tensor = unpadded_tensor[:, :MAX_FRAME_PER_BATCH]
        ground_truth_batch_frame = np.zeros((data_batch.shape[0], PIANO_PITCHES))
        ground_truth_batch_frame[:unpadded_tensor.shape[1], :unpadded_tensor.shape[2]] = unpadded_tensor[0]
        ground_truth_batch_onset = np.zeros((data_batch.shape[0], PIANO_PITCHES))
        ground_truth_batch_onset[:unpadded_tensor.shape[1], :unpadded_tensor.shape[2]] = unpadded_tensor[1]
        return data_batch, ground_truth_batch_frame, ground_truth_batch_onset


if __name__ == '__main__':
    init()
    with ZipFile("../data/MAPS/MAPS_AkPnBcht_1.zip") as zipfile:
        cqt, sr = wav_to_CQT(zipfile.open("AkPnBcht/UCHO/I32-96/C0-5-9/MAPS_UCHO_C0-5-9_I32-96_S0_n13_AkPnBcht.wav"))
        display_CQT(cqt, sr)

        midi_tensor = midi_file_to_tensor(
            zipfile.open("AkPnBcht/UCHO/I32-96/C0-5-9/MAPS_UCHO_C0-5-9_I32-96_S0_n13_AkPnBcht.mid"))
        show_images = True
        do_image(((midi_tensor[0] > 0).astype(float) + (midi_tensor[1] > 0)) / 2, title="debug", folder=PATH_DEBUG)
    compare_midi(PATH_DEBUG + "bug.mid", PATH_DEBUG + "bug.mid", PATH_DEBUG + "test.PNG")
