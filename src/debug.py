#!/usr/bin/python3.5

from config import *
from init import init
from data_utils import *
from utils import do_image

TEST_PATHS = []
TRAIN_PATHS = []

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

def dataset_vis(music_pair, debug_dir):
    """ Finds a music, puts it in debug_dir (midi and wav) and creates ground_truth image.

    :param music_pair: music to find in the form (path_to_zip, path_from_zip_to_music_without_extension).
    :param debug_dir: directory to put every output files.
    """
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    music_name = music_pair[1].split("/")[-1]
    data_batch, ground_truth_batch_frame, ground_truth_batch_onset = util_next_batch(music_name=music_name)
    to_img = ((ground_truth_batch_onset > 0).astype(int) + (ground_truth_batch_frame > 0)) / 2
    to_img = np.append(to_img, data_batch[:, :, 0]*5, axis=1)
    do_image(to_img, "Ground_Truth_Onset_And_frame", debug_dir)
    with ZipFile(music_pair[0]) as zipfile:
        zipfile.extract(music_pair[1] + ".wav", debug_dir)
        zipfile.extract(music_pair[1] + ".mid", debug_dir)


if __name__ == "__main__":
    init()
    init_path_lists()
    for i in range(len(TRAIN_PATHS)):
        # if 'ENS' in TEST_PATHS[i][0]:
        dataset_vis(TRAIN_PATHS[i], PATH_DEBUG + "test_" + str(i) + "/")
