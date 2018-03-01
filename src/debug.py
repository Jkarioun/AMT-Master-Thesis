from config import *
from init import init
from data_utils import *
from utils import do_image

#find a music, put it in data/debug (midi and wav) and create ground_truth image.
def dataset_vis(music_pair, debug_dir):
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    data_batch, ground_truth_batch_frame, ground_truth_batch_onset = util_next_batch(music_pair=music_pair)
    to_img = ((ground_truth_batch_onset.T > 0).astype(int) + (ground_truth_batch_frame.T > 0)) / 2
    to_img = np.append(to_img.T, data_batch[:, :, 0], axis=1)
    do_image(to_img.T, "Ground_Truth_Onset_And_frame", debug_dir, visualization_path=False)
    with ZipFile(music_pair[0]) as zipfile:
        outpath = debug_dir
        zipfile.extract(music_pair[1]+".wav", outpath)
        zipfile.extract(music_pair[1]+".mid", outpath)


if __name__=="__main__":
    init()
    for i in range(len(TEST_PATHS)):
        if 'ENS' in TEST_PATHS[i][0]:
            dataset_vis(TEST_PATHS[i], PATH_DEBUG+"test_"+str(i))

