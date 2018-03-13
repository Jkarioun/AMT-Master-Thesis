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
            'kelz_batch_norm_test'
        ]

        attributes = [
            ('log_loss', float),
            ('accuracy', float),
            ('TP', float),
            ('TN', float),
            ('FP', float),
            ('FN', float),
            ('TP_mod', float),
            ('TN_mod', float),
            ('FP_mod', float),
            ('FN_mod', float),
            ('iteration', int),
            ('rand_seed', int),
            ('mode', str),
        ]

        train_log_loss = [[] for _ in range(len(models))]

        test_same = [{name: [] for name, _ in attributes} for _ in range(len(models))]
        test_changing = [{name: [] for name, _ in attributes} for _ in range(len(models))]

        patterns = {
            name: re.compile('\[%s=([\w\.]+)\]' % name)
            for (name, _) in attributes
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
                    mode = attr(line, 'mode', str)
                    rand_seed = attr(line, 'rand_seed', int)

                    # train
                    if mode == 'train':
                        train_log_loss[index].append(math.log(attr(line, 'log_loss', float)))
                    # test
                    # same test data
                    elif rand_seed == RAND_SEED:
                        for name, type in attributes:
                            test_same[index][name].append(attr(line, name, type))
                    # random test data
                    else:
                        for name, type in attributes:
                            test_changing[index][name].append(attr(line, name, type))

                    line_num += 1

        for model, index in zip(models, range(len(models))):
            model_mva = [sum(train_log_loss[index][i:i + mva_length]) / mva_length for i in
                         range(len(train_log_loss[index]) - mva_length)]

            plt.plot(range(len(model_mva)), model_mva, '-')

        plt.legend(models, loc='upper right')
        plt.ylabel('Log loss (moving average over 300 steps)')
        plt.yscale('log')
        plt.xlabel('Steps (5 iterations per steps)')
        plt.title('Train loss evolution over training')

        plt.show()

        for model, index in zip(models, range(len(models))):
            plt.plot(range(len(test_same[index]['log_loss'])), test_same[index]['log_loss'], '-')

        plt.legend(models, loc='upper right')
        plt.ylabel('Log loss')
        plt.yscale('log')
        plt.xlabel('Steps (50 training iterations per step)')
        plt.title('Test loss evolution over training (same test data)')

        plt.show()

        for model, index in zip(models, range(len(models))):
            plt.plot(range(len(test_changing[index]['log_loss'])), test_changing[index]['log_loss'], '-')

        plt.legend(models, loc='upper right')
        plt.ylabel('Log loss')
        plt.yscale('log')
        plt.xlabel('Steps (50 training iterations per step)')
        plt.title('Test loss evolution over training (random test data)')

        plt.show()

        # F-measure
        mva_length = 10
        for model, index in zip(models, range(len(models))):
            TPs = test_same[index]['TP']
            FPs = test_same[index]['FP']
            TNs = test_same[index]['TN']
            FNs = test_same[index]['FN']

            precision_mva = [sum(TPs[i:i + mva_length]) / (sum(TPs[i:i + mva_length]) + sum(FPs[i:i + mva_length])
                                                           + 0.0000001)
                          for i in range(len(TPs) - mva_length)]
            recall_mva = [sum(TPs[i:i + mva_length]) / (sum(TPs[i:i + mva_length]) + sum(FNs[i:i + mva_length])
                                                        + 0.0000001)
                          for i in range(len(TPs) - mva_length)]
            F_measure_mva = [2 * precision_mva[i] * recall_mva[i] / (precision_mva[i] + recall_mva[i]
                                                                     + 0.0000001)
                          for i in range(len(precision_mva))]

            plt.plot(range(len(F_measure_mva)), F_measure_mva, '-')

        plt.legend(models, loc='lower right')
        plt.ylabel('F-measure')
        plt.xlabel('Steps (50 training iterations per step)')
        plt.title('F-measure evolution over training (same test data)')

        plt.show()
