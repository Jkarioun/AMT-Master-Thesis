#!/usr/bin/python3.5

from config import *
from init import init
from data_utils import *
from utils import do_image


#
def dataset_vis(music_pair, debug_dir):
    """ Finds a music, puts it in debug_dir (midi and wav) and creates ground_truth image.

    :param music_pair: music to find in the form (path_to_zip, path_from_zip_to_music_without_extension).
    :param debug_dir: directory to put every output files.
    """
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    data_batch, ground_truth_batch_frame, ground_truth_batch_onset = util_next_batch(music_pair=music_pair)
    to_img = ((ground_truth_batch_onset.T > 0).astype(int) + (ground_truth_batch_frame.T > 0)) / 2
    to_img = np.append(to_img.T, data_batch[:, :, 0], axis=1)
    do_image(to_img, "Ground_Truth_Onset_And_frame", debug_dir)
    with ZipFile(music_pair[0]) as zipfile:
        zipfile.extract(music_pair[1] + ".wav", debug_dir)
        zipfile.extract(music_pair[1] + ".mid", debug_dir)


if __name__ == "__main__":
    init()
    for i in range(len(TRAIN_PATHS)):
        # if 'ENS' in TEST_PATHS[i][0]:
        dataset_vis(TRAIN_PATHS[i], PATH_DEBUG + "test_" + str(i) + "/")
