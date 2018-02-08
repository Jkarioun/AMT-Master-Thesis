from config import *


def init():

    # Hack to handle large MIDI files
    pm.pretty_midi.MAX_TICK = 1e10

    # Seeds
    np.random.seed(101)
    tf.set_random_seed(101)

    for i in range(TOTAL_BIN):
        for j, j_idx in zip(BINS_PER_PITCH * HARMONIC_RELATIVES, range(HARMONIC_RELATIVES.size)):
            if 0 <= i + j < TOTAL_BIN:
                HARMONIC_MAPPING[i * HARMONIC_RELATIVES.size + j_idx] = i + j
            else:
                HARMONIC_MAPPING[i * HARMONIC_RELATIVES.size + j_idx] = TOTAL_BIN
