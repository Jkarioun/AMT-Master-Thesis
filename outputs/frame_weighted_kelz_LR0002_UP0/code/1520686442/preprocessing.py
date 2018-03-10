from config import *
from data_utils import wav_to_CQT

print(os.getcwd())

for zipfile_name in [f for f in os.listdir(PATH_MAPS) if f.endswith('.zip')]:
    with ZipFile(PATH_MAPS + zipfile_name) as zipfile:
        for path in zipfile.namelist():
            # skip folders
            if not os.path.basename(path):
                continue
            file_data = path.split('/')

            # convert wav to npy files
            if len(file_data) > 1 and file_data[-1][-3:] == 'wav':
                # save tensor
                tensor, _ = wav_to_CQT(zipfile.open(path))

                # cut if too long
                if tensor.shape[0] > MAX_FRAME_PER_BATCH:
                    tensor = tensor[:MAX_FRAME_PER_BATCH]
                np.save(PATH_MAPS_PREPROCESSED + file_data[-1][:-4] + '.npy', tensor)
            # save txt and mid files
            else:
                source = zipfile.open(path)
                target = open(PATH_MAPS_PREPROCESSED + file_data[-1], 'wb')
                with source, target:
                    shutil.copyfileobj(source, target)

