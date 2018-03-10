from config import *
from data_utils import wav_to_CQT
from time import time

for zipfile_name in [f for f in os.listdir(PATH_MAPS) if f.endswith('.zip')]:
    with ZipFile(PATH_MAPS + zipfile_name) as zipfile:
        for path in zipfile.namelist():
            file_data = path.split('/')

            # skip folders
            if not os.path.basename(path) or 'readme' in file_data[-1]:
                continue

            # convert wav to npy files
            if len(file_data):
                extension = file_data[-1][-3:]
                name = file_data[-1][:-4]
                if extension == 'wav':
                    # save tensor
                    tensor, _ = wav_to_CQT(zipfile.open(path))

                    # cut if too long
                    if tensor.shape[0] > MAX_FRAME_PER_BATCH:
                        tensor = tensor[:MAX_FRAME_PER_BATCH]
                    np.save(PATH_MAPS_PREPROCESSED + name + '.npy', tensor)
                # save txt and mid files
                if extension == 'mid':
                    source = zipfile.open(path)
                    target = open(PATH_MAPS_PREPROCESSED + name + '.mid', 'wb')
                    with source, target:
                        shutil.copyfileobj(source, target)



