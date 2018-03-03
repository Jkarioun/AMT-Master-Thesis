from config import *


def do_image(data, title, folder, visualization_path=True):
    im = Image.new("L", (len(data[0]), len(data)))
    im.putdata([item * 255 for sublist in data[::-1] for item in sublist])
    im.save((PATH_VISUALISATION if visualization_path else "") + folder + '/' + title + '.png')
    # plt.savefig(PATH_VISUALISATION + folder + '/' + title + '.png')
    if show_images:
        plt.pcolormesh(data)
        plt.title(title)
        plt.show()


def accuracy(ground_truth, pred):
    return np.mean(ground_truth == pred)
