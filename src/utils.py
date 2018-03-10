#!/usr/bin/python3.5

from config import *


def onsets_and_frames_to_weights(onsets, frames, onset):
    """ Computes loss weights from the onsets and the frames ground truth.

    :param onsets: onset ground_truth.
    :param frames: frame ground_truth.
    :param onset: True if the loss is relative to the onsets, False if it is relative to the frames.
    :return: A [frame, pitch] numpy array to weight the loss.
    """
    assert (onsets.shape == frames.shape)
    onsets = onsets.T
    frames = frames.T

    output = np.ones(onsets.shape, float)

    if onset:
        for pitch in range(len(onsets)):
            for frame in range(len(onsets[pitch])):
                if onsets[pitch, frame] > 0:
                    output[pitch, frame] = 100
        return output.T

    for pitch in range(len(onsets)):
        time_from_onset = 10000
        for frame in range(len(onsets[pitch])):
            if onsets[pitch, frame] > 0:
                time_from_onset = 0
                output[pitch, frame] = 0  # onset could be missplaced
                if frame - 1 >= 0:
                    output[pitch, frame - 1] = 0  # onset could be missplaced
            if (time_from_onset >= MIN_NOTE_LENGTH_IF_SUSTAIN and frames[pitch, frame] > 0) or (
                    time_from_onset <= MAX_NOTE_LENGTH and frames[pitch, frame] == 0):
                # the note might not make any sound anymore or may be continuing through sustain pedal.
                output[pitch, frame] = 0
            elif frames[pitch, frame] > 0:
                output[pitch, frame] = 10
            else:
                output[pitch, frame] = 1
            time_from_onset += 1
    return output.T


def do_image(data, title, folder):
    """ Create an image from data for better analysis.

    :param data: [frame, pitch] numpy array to visualise, with values from 0 to 1.
    :param title: title to give to the image and the file containing it.
    :param folder: folder to put the image in.
    """

    assert folder[-1] == "/", "Folder path should end with a /"
    im = Image.new("L", data.shape)
    im.putdata([item * 255 for sublist in data.T[::-1] for item in sublist])
    im.save(folder + title + '.png')
    if show_images and DISPLAY:
        plt.pcolormesh(data.T)
        plt.title(title)
        plt.show()


def testing_metrics(ground_truth, pred, weight=None):
    """ Computes the TP, FP, FN and TN of the prediction.

    :param ground_truth: boolean tensor of the form [frame, pitch] representing the ground_truth.
    :param pred: boolean tensor with same dimensions as ground_truth representing the prediction made.
    :return: a dictionary with 4 integers: {'TP':#TP, 'FP':#FP, 'FN':#FN, 'TN':#TN} of the prediction.
    """

    la = np.logical_and

    if weight is None:
        return {'TP': np.sum(la(ground_truth, pred)),
                'FP': np.sum(la(np.logical_not(ground_truth), pred)),
                'FN': np.sum(la(ground_truth, np.logical_not(pred))),
                'TN': np.sum(la(np.logical_not(ground_truth), np.logical_not(pred)))}

    return {'TP': np.sum(la(la(ground_truth, pred), weight)),
            'FP': np.sum(la(la(np.logical_not(ground_truth), pred), weight)),
            'FN': np.sum(la(la(ground_truth, np.logical_not(pred)), weight)),
            'TN': np.sum(la(la(np.logical_not(ground_truth), np.logical_not(pred)), weight))}


def accuracy(ground_truth, pred):
    return np.mean(ground_truth == pred)


if __name__ == "__main__":
    data = np.linspace(0, 1, 10)
    data = data.reshape((2, 5))
    print(data)
    do_image(data, "test", PATH_DEBUG)
