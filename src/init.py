from config import *
from zipfile import ZipFile
import os


def init():
    print("Starting initialization.")

    # Hack to handle large MIDI files
    pm.pretty_midi.MAX_TICK = 1e10

    # Seeds
    np.random.seed(42)
    tf.set_random_seed(42)


if __name__ == '__main__':
    for zipfile_name in [f for f in os.listdir(PATH_MAPS) if f[-3:] == 'zip']:
        with ZipFile(PATH_MAPS + zipfile_name) as zipfile:
            pass
