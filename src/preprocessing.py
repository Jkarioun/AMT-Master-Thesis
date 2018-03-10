#!/usr/bin/python3.5

from config import *
from data_utils import wav_to_CQT

if not os.path.exists(PATH_MAPS_PREPROCESSED):
    os.makedirs(PATH_MAPS_PREPROCESSED)

total_i = len([f for f in os.listdir(PATH_MAPS) if f.endswith('.zip')])
i = 0

for zipfile_name in [f for f in os.listdir(PATH_MAPS) if f.endswith('.zip')]:
    with ZipFile(PATH_MAPS + zipfile_name) as zipfile:
        i += 1
        total_j = len(zipfile.namelist())
        j = 0
        for path in zipfile.namelist():
            j += 1
            print(i, "/", total_i, "\t \t", j, "/", total_j)

            # convert wav to npy files
            if path.endswith(".wav"):
                name = path.split('/')[-1][:-4]
                # compute and save CQT
                tensor, _ = wav_to_CQT(zipfile.open(path))
                if tensor.shape[0] > MAX_FRAME_PER_BATCH:
                    tensor = tensor[:MAX_FRAME_PER_BATCH]
                np.save(PATH_MAPS_PREPROCESSED + name + '.npy', tensor)

            # save txt and mid files
            if path.endswith(".mid"):
                name = path.split('/')[-1][:-4]
                with zipfile.open(path) as source, \
                        open(PATH_MAPS_PREPROCESSED + name + '.mid', 'wb') as target:
                    shutil.copyfileobj(source, target)
