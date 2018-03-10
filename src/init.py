#!/usr/bin/python3.5

from config import *


def init():
    """ Initialises config constant lists, creates output folders, ... ."""
    print("Starting initialization.")

    # Hack to handle large MIDI files
    pm.pretty_midi.MAX_TICK = 1e10

    # Seeds
    np.random.seed(42)
    tf.set_random_seed(42)

    init_path_lists()
    create_folders()
    # Logger config
    logging.basicConfig(filename=PATH_LOGS + CONFIG_NAME + '.log', level=logging.DEBUG,
                        format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')


def init_path_lists():
    """ Initialises the training and testing path list. """

    for filename in os.listdir(PATH_MAPS_PREPROCESSED):
        if filename.endswith('.npy') and (USE_ENSTDk or 'ENSTDk' not in filename):
            name = filename[:-4]
            assert name + ".mid" in os.listdir(PATH_MAPS_PREPROCESSED),\
                name + ".npy haven't any corresponding midi file."
            if name.startswith('MAPS_MUS-'):
                TEST_FILENAMES.append(name)
            else:
                TRAIN_FILENAMES.append(name)


def create_folders():
    """ Creates the output folders and save the used code if necessary. """

    def copy(src, dest):
        try:
            shutil.copytree(src, dest)
        except OSError as e:
            # If the error was caused because the source wasn't a directory
            if e.errno == errno.ENOTDIR:
                shutil.copy(src, dest)
            else:
                print('Directory not copied. Error: %s' % e)

    assert (not TRAIN_FROM_LAST or os.path.exists(PATH_OUTPUT)), \
        "Impossible to train from last version: no last version available for this config_name"
    assert (TRAIN_FROM_LAST or not os.path.exists(PATH_OUTPUT)), \
        "config_name already used for another model. Please change the name or suppress the output folder."
    if not TRAIN_FROM_LAST:
        os.makedirs(PATH_VISUALISATION)
        os.makedirs(PATH_LOGS)
        os.makedirs(PATH_TENSORBOARD)
        os.makedirs(PATH_CHECKPOINTS)
        copy(".", PATH_CODE)
    else:
        copy(".", PATH_CODE + str(int(time.time())) + "/")


if __name__ == '__main__':
    init()
