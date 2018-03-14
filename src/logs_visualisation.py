#!/usr/bin/python3.5

from config import *

if DISPLAY:
    import math
    import matplotlib.pyplot as plt
    import re

    if __name__ == '__main__':

        # Test moving average length
        mva_length = 500

        # Models to compare
        models = [
            'onset_weighted_hybrid_LR0002_UP0',
            'onset_weighted_harm_LR0002_UP0',
            'onset_weighted_kelz_LR0002_UP0',
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
                        train_log_loss[index].append(attr(line, 'log_loss', float))

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
        mva_length = 20
        precision_mva = []
        recall_mva = []
        F_measure_mva = []

        for mod in ['', '_mod']:
            for index in range(len(models)):
                TPs = test_same[index]['TP'+mod]
                FPs = test_same[index]['FP'+mod]
                TNs = test_same[index]['TN'+mod]
                FNs = test_same[index]['FN'+mod]

                sum_TP = [sum(TPs[i:i + mva_length]) for i in range(len(TPs) - mva_length)]
                sum_FP = [sum(FPs[i:i + mva_length]) for i in range(len(TPs) - mva_length)]
                sum_FN = [sum(FNs[i:i + mva_length]) for i in range(len(TPs) - mva_length)]
                precision_mva.append([sum_TP[i] / (sum_TP[i] + sum_FP[i]) if sum_TP[i] > 0 else 0 for i in range(len(sum_TP))])
                recall_mva.append([sum_TP[i] / (sum_TP[i] + sum_FN[i]) if sum_TP[i] > 0 else 0 for i in range(len(sum_TP))])
                F_measure_mva.append([2 * precision_mva[-1][i] * recall_mva[-1][i] /
                                      (precision_mva[-1][i] + recall_mva[-1][i]) if precision_mva[-1][i] > 0 else 0
                                      for i in range(len(precision_mva[-1]))])

            for index in range(len(models)):
                plt.plot(range(len(F_measure_mva[index])), F_measure_mva[index], '-')
            plt.legend(models, loc='lower right')
            plt.ylabel('F-measure'+mod)
            plt.xlabel('Steps (50 training iterations per step)')
            plt.title('F-measure evolution over training (same test data)')
            plt.show()

            for index in range(len(models)):
                plt.plot(range(len(precision_mva[index])), precision_mva[index], '-')
            plt.legend(models, loc='lower right')
            plt.ylabel('Precision'+mod)
            plt.xlabel('Steps (50 training iterations per step)')
            plt.title('Precision evolution over training (same test data)')
            plt.show()

            for index in range(len(models)):
                plt.plot(range(len(recall_mva[index])), recall_mva[index], '-')
            plt.legend(models, loc='lower right')
            plt.ylabel('Recall'+mod)
            plt.xlabel('Steps (50 training iterations per step)')
            plt.title('Recall evolution over training (same test data)')
            plt.show()
