from config import *


def init():
    print("Starting initialization.")

    # Hack to handle large MIDI files
    pm.pretty_midi.MAX_TICK = 1e10

    # Seeds
    np.random.seed(42)
    tf.set_random_seed(42)

    init_path_lists()


def append_to(a_list, zipfile_name, path):
    if path.endswith(".mid"):
        a_list.append([zipfile_name, path[:-4]])
    elif path.endswith(".wav"):
        assert path[:-4] == a_list[-1][1], "non-corresponding files"


def init_path_lists():
    for zipfile_name in [f for f in os.listdir(PATH_MAPS) if f.endswith('.zip')]:
        with ZipFile(PATH_MAPS + zipfile_name) as zipfile:
            for path in zipfile.namelist():
                if len(path.split("/")) > 1:
                    if path.split("/")[1] == 'MUS':
                        append_to(TEST_PATHS, PATH_MAPS + zipfile_name, path)
                    else:
                        append_to(TRAIN_PATHS, PATH_MAPS + zipfile_name, path)


if __name__ == '__main__':
    init()
    pass
