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

    INPUT_HEIGHT = 3
    INPUT_WIDTH = 3

    music_name = music_pair[1].split("/")[-1]
    data_batch, ground_truth_batch_frame, ground_truth_batch_onset = util_next_batch(music_name=music_name)

    ground_truth_image = ((ground_truth_batch_onset > 0).astype(int) + (ground_truth_batch_frame > 0)) / 2
    dim1 = ground_truth_image.shape
    input_image = np.power(data_batch[:, :, 0] / np.max(data_batch), 0.5)/255*200
    dim2 = input_image.shape

    img = np.array([[input_image[frame//INPUT_WIDTH, pitch_pix//INPUT_HEIGHT]*255 for pitch_pix in range(INPUT_HEIGHT*dim2[1])] for frame in range(INPUT_WIDTH*dim2[0])])

    for frame in range(INPUT_WIDTH*len(ground_truth_image)):
        for pitch in range(len(ground_truth_image[frame//INPUT_WIDTH])):
            if ground_truth_image[frame//INPUT_WIDTH, pitch] > 0:
                pitch_pos = pitch*BINS_PER_PITCH*INPUT_HEIGHT+(BINS_PER_PITCH*INPUT_HEIGHT)//2
                img[frame, pitch_pos] = 200 + 55 * ground_truth_image[frame//INPUT_WIDTH, pitch]
                img[frame, pitch_pos-1] = 200 + 55 * ground_truth_image[frame//INPUT_WIDTH, pitch]

    im = Image.new('L', (INPUT_WIDTH*dim2[0], INPUT_HEIGHT*dim2[1]))
    im.putdata([255-item for sublist in img.T[::-1] for item in sublist])
    im.save(debug_dir + 'input_vs_GT' + '.png')

    do_image(ground_truth_image, "Ground_Truth_Onset_And_frame", debug_dir)
    with ZipFile(music_pair[0]) as zipfile:
        zipfile.extract(music_pair[1] + ".wav", debug_dir)
        zipfile.extract(music_pair[1] + ".mid", debug_dir)


if __name__ == "__main__":
    init()
    init_path_lists()
    for i in range(len(TRAIN_PATHS)):
        # if 'ENS' in TEST_PATHS[i][0]:
        dataset_vis(TRAIN_PATHS[i], PATH_DEBUG + "test_" + str(i) + "/")
