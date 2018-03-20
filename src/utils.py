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


def moving_average(vector, mva_length, average=True):
    ''' Compute the moving average of the vector

    :param vector: vector from which to compute the moving average.
    :param mva_length: length of the moving average window.
    :param average: False if sum is wanted instead of average.
    :return: numpy vector of the moving average of length (len(vector)-mva_length+1)
    '''
    cumsum = np.insert(np.cumsum(vector), 0, 0)
    if average:
        return (cumsum[mva_length:] - cumsum[:-mva_length]) / mva_length
    return (cumsum[mva_length:] - cumsum[:-mva_length])


def AUC(pred, truth, x_points=None):
    """ Computes the AUC of the prediction.

    :param pred: real-valued prediction (highest is activated) of a sample in the form of a vector.
    :param truth: ground truth (1 is activated) of a sample in the form of a vector.
    :param x_points: make x_points points for the graph. None for making all points.
    :return: the AUC score, and the x and y vectors to plot the ROC or None and None if x_points=0.
    """
    assert len(pred) == len(truth)

    _, truth = (np.array(t) for t in zip(*sorted(zip(pred, truth), key=lambda x: -x[0])))
    cum_truth = np.cumsum(truth)
    AUC_score = np.sum(cum_truth[truth == 0]) / np.sum(truth == 1) / np.sum(truth == 0)

    # Graph part
    if x_points == 0:
        return AUC_score, None, None
    if x_points is None:
        x_points = len(pred) + 1
    points_taken = np.linspace(0, len(pred), x_points, dtype=int)
    y = np.insert(cum_truth, 0, 0)[points_taken]
    x = points_taken - y
    return (AUC_score), x, y


if __name__ == "__main__":
    score, x, y = AUC([0, 1, 10, 2, 3, 9, 7], [0, 0, 1, 1, 0, 0, 1], 3)
    print(score)
    plt.plot(x, y, '-')
    plt.show()
    data = np.linspace(0, 1, 10)
    data = data.reshape((2, 5))
    print(data)
    do_image(data, "test", PATH_DEBUG)
