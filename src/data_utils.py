from config import *
from init import init


def wav_to_CQT(file):
    tmp = io.BytesIO(file.read())
    y, sr = sf.read(tmp)
    y = librosa.core.to_mono(y.T)
    y = librosa.core.resample(y, orig_sr=sr, target_sr=SAMPLE_RESOLUTION)
    sr = SAMPLE_RESOLUTION
    cqt = abs(librosa.core.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, fmin=MIN_FREQ, n_bins=NUM_PITCHES * BINS_PER_PITCH,
                               bins_per_octave=BINS_PER_OCTAVE, filter_scale=1, norm=1, sparsity=0, window=CQT_WINDOW,
                               scale=True, pad_mode='constant'))
    return cqt, sr


def display_CQT(cqt, sr):
    librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()


def compare_midi(gold_MIDI_path, pred_MIDI_path, output_path):
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


def midi_file_to_tensor(file, onset=False):
    """ Returns the information of the midi file in the form
    output[frame,pitch] = onset_during_this_frame if onset else pitch_played_during_this_frame
    """
    midi = pm.PrettyMIDI(io.BytesIO(file.read()))
    frames = math.floor(midi.get_end_time() * FRAME_PER_SEC)
    if onset:
        output = np.full((frames, PIANO_PITCHES), fill_value=False, dtype=bool)
        for note in midi.instruments[0].notes:
            output[math.floor(note.start * FRAME_PER_SEC), note.pitch - PIANO_MIN_PITCH] = True
    else:
        output = midi.get_piano_roll(FRAME_PER_SEC)[PIANO_MIN_PITCH:PIANO_MAX_PITCH + 1, :].T > 0
    return output


def next_batch(i, train=True, onset=False):
    np.random.seed(i)
    tf.set_random_seed(i)
    data_batch, ground_truth_batch = util_next_batch(train=train, onset=onset)
    while data_batch.shape[0] < 1000:
        inputs, outputs = util_next_batch(train=train, onset=onset)
        data_batch = np.concatenate((data_batch, inputs))
        ground_truth_batch = np.concatenate((ground_truth_batch, outputs))

    if data_batch.shape[0] > 2000:
        data_batch = data_batch[:2000]
        ground_truth_batch = ground_truth_batch[:2000]

    return [data_batch], [ground_truth_batch]


def util_next_batch(train=True, onset=False):
    """ Returns the next batch for training.
    """
    if train:
        pair = TRAIN_PATHS[np.random.randint(0, len(TRAIN_PATHS))]
    else:
        pair = TEST_PATHS[np.random.randint(0, len(TEST_PATHS))]
    with ZipFile(pair[0]) as zipfile:
        # input
        print(pair[0]+"   "+pair[1])
        data_batch, _ = wav_to_CQT(zipfile.open(pair[1] + ".wav"))
        data_batch = np.reshape(data_batch, [-1, TOTAL_BIN, 1])
        # expected output
        unpadded_tensor = midi_file_to_tensor(zipfile.open(pair[1] + ".mid"), onset=onset)
        ground_truth_batch = np.zeros((data_batch.shape[0], PIANO_PITCHES))
        ground_truth_batch[:unpadded_tensor.shape[0], :unpadded_tensor.shape[1]] = unpadded_tensor
    return data_batch, ground_truth_batch


if __name__ == '__main__':
    init()
    with ZipFile("../data/MAPS/MAPS_AkPnBcht_1.zip") as zipfile:
        cqt, sr = wav_to_CQT(zipfile.open("AkPnBcht/UCHO/I32-96/C0-5-9/MAPS_UCHO_C0-5-9_I32-96_S0_n13_AkPnBcht.wav"))
        display_CQT(cqt, sr)

        midi_tensor = midi_file_to_tensor(zipfile.open("AkPnBcht/UCHO/I32-96/C0-5-9/MAPS_UCHO_C0-5-9_I32-96_S0_n13_AkPnBcht.mid"), onset=False)
        plt.imshow(midi_tensor)
        plt.show()
    compare_midi(PATH_DEBUG + "bug.mid", PATH_DEBUG + "bug.mid", PATH_VISUALISATION + "test.PNG")

