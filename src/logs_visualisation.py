#!/usr/bin/python3.5

from config import *

if DISPLAY:
    import math
    import matplotlib.pyplot as plt
    import re

    if __name__ == '__main__':

        # Test moving average length
        mva_length = 300

        # Models to compare
        models = [
            'frame_weighted_hybrid_0002',
            'frame_weighted_hybrid37_LR0002_UP0'
        ]

        train_log_loss = [[] for _ in range(len(models))]
        test_log_loss_same = [[] for _ in range(len(models))]
        test_log_loss_changing = [[] for _ in range(len(models))]

        patterns = {
            'iteration': re.compile('\[iteration=(\d+)\]'),
            'TP': re.compile('\[TP=(\d+)\]'),
            'TN': re.compile('\[TN=(\d+)\]'),
            'FP': re.compile('\[FP=(\d+)\]'),
            'FN': re.compile('\[FN=(\d+)\]'),
            'log_loss': re.compile('\[log_loss=(.*)\]'),
            'rand_seed': re.compile('\[rand_seed=(\d+)\]'),
            'accuracy': re.compile('\[accuracy=(.*)\]'),
            'measure': re.compile('\[measure=(\w+)\]'),
            'mode': re.compile('\[mode=(\w+)\]')
        }


        def attr(line, key, cast_funct):
            """ Herlper to get an attribute from a line of log
                :param line: line of log to be treated
                :param key: name of the attribute to lookup
                :param cast_funct: function to use to get the appropriate type"""
            res = re.search(patterns[key], line)
            if res is not None:
                return cast_funct(res.groups(1)[0])
            return None


        for model, index in zip(models, range(len(models))):
            with open('../outputs/{0}/logs/{0}.log'.format(model)) as log_file:
                iteration = 0

                line_iter = 0
                line_iter_test = 0

                line_num = 0
                for line in log_file:
                    iteration = attr(line, 'iteration', str)

                    # skip invalid lines
                    if iteration is None:
                        continue

                    # get attr values
                    log_loss = attr(line, 'log_loss', float)
                    mode = attr(line, 'mode', str)
                    rand_seed = attr(line, 'rand_seed', int)

                    # train
                    if mode == 'train':
                        train_log_loss[index].append(math.log(log_loss))
                    # test
                    # same test data
                    elif rand_seed == RAND_SEED:
                        test_log_loss_same[index].append(math.log(log_loss))
                    # random test data
                    else:
                        test_log_loss_changing[index].append(math.log(log_loss))

                    line_num += 1

        for model, index in zip(models, range(len(models))):
            model_mva = [sum(train_log_loss[index][i:i + mva_length]) / mva_length for i in
                              range(len(train_log_loss[index]) - mva_length)]

            plt.plot(range(len(model_mva)), model_mva, '-')

        plt.legend(models, loc='upper right')
        plt.ylabel('Log loss (moving average over 300 steps)')
        plt.xlabel('Steps (5 iterations per steps)')
        plt.title('Train loss evolution over training')

        plt.show()

        for model, index in zip(models, range(len(models))):
            plt.plot(range(len(test_log_loss_same[index])), test_log_loss_same[index], '-')

        plt.legend(models, loc='upper right')
        plt.ylabel('Log loss')
        plt.xlabel('Steps (500 training iterations per step)')
        plt.title('Test loss evolution over training (same test data)')

        plt.show()

        for model, index in zip(models, range(len(models))):
            plt.plot(range(len(test_log_loss_same[index])), test_log_loss_same[index], '-')

        plt.legend(models, loc='upper right')
        plt.ylabel('Log loss')
        plt.xlabel('Steps (500 training iterations per step)')
        plt.title('Test loss evolution over training (random test data)')

        plt.show()
