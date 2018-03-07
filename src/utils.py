from config import *


def onsets_and_frames_to_weights(onsets, frames, onset):
    assert (onsets.shape == frames.shape)
    onsets = onsets.T
    frames = frames.T

    output = np.ones(onsets.shape, float)

    if onset:
        for pitch in range(len(onsets)):
            for time in range(len(onsets[pitch])):
                if onsets[pitch, time] > 0:
                    output[pitch, time] = 100
        return output.T

    for pitch in range(len(onsets)):
        time_from_onset = 10000
        for time in range(len(onsets[pitch])):
            if onsets[pitch, time] > 0:
                time_from_onset = 0
                output[pitch, time] = 0  # onset could be missplaced
                if time - 1 >= 0:
                    output[pitch, time - 1] = 0  # onset could be missplaced
            if (time_from_onset >= TOO_LONG_MIN and frames[pitch, time] > 0) or (
                    time_from_onset <= TOO_LONG_MAX and frames[pitch, time] == 0):
                # the note might not make any sound anymore or may be continuing through sustain pedal.
                output[pitch, time] = 0
            elif frames[pitch, time] > 0:
                output[pitch, time] = 10
            else:
                output[pitch, time] = 1
            time_from_onset += 1
    return output.T


def do_image(data, title, folder, visualization_path=True):
    """

    :param data: [frame, pitch]
    :param title:
    :param folder:
    :param visualization_path:
    :return:
    """
    data = data.T
    im = Image.new("L", (len(data[0]), len(data)))
    im.putdata([item * 255 for sublist in data[::-1] for item in sublist])
    im.save((PATH_VISUALISATION if visualization_path else "") + folder + '/' + title + '.png')
    if show_images and DISPLAY:
        plt.pcolormesh(data)
        plt.title(title)
        plt.show()


def accuracy(ground_truth, pred):
    return np.mean(ground_truth == pred)
