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

    def append_to(a_list, zipfile_name, path):
        if path.endswith(".mid"):
            a_list.append([zipfile_name, path[:-4]])
        elif path.endswith(".wav"):
            assert path[:-4] == a_list[-1][1], "non-corresponding files"

    for zipfile_name in [f for f in os.listdir(PATH_MAPS) if f.endswith('.zip')]:
        with ZipFile(PATH_MAPS + zipfile_name) as zipfile:
            for path in zipfile.namelist():
                if len(path.split("/")) > 1 and (USE_ENSTDk or not path[:3] == "ENS"):
                    if path.split("/")[1] == 'MUS':
                        append_to(TEST_PATHS, PATH_MAPS + zipfile_name, path)
                    else:
                        append_to(TRAIN_PATHS, PATH_MAPS + zipfile_name, path)


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

    assert (not TRAIN_FROM_LAST or os.path.exists(PATH_OUTPUT)),\
        "Impossible to train from last version: no last version available for this config_name"
    assert (TRAIN_FROM_LAST or not os.path.exists(PATH_OUTPUT)),\
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
