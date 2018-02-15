from config import *


def init():
    print("Starting initialization.")

    # Hack to handle large MIDI files
    pm.pretty_midi.MAX_TICK = 1e10

    # Seeds
    np.random.seed(42)
    tf.set_random_seed(42)