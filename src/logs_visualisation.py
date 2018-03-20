#!/usr/bin/python3.5

from config import *

if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    import re
    from utils import moving_average

    # Test moving average length
    MVA_LENGTH_TRAIN = 500
    MVA_LENGTH_TEST_SAME = 30
    MVA_LENGTH_TEST_CHANGING = 30

    # Models to compare
    models = [out for out in os.listdir('../outputs') if out.startswith('onset')]

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

    train_log_loss_iteration = [[] for _ in range(len(models))]
    train_log_loss = [[] for _ in range(len(models))]

    test_same_iteration = [[] for _ in range(len(models))]
    test_same = [{name: [] for name, _ in attributes} for _ in range(len(models))]
    test_changing_iteration = [[] for _ in range(len(models))]
    test_changing = [{name: [] for name, _ in attributes} for _ in range(len(models))]

    patterns = {name: re.compile('\[%s=([\w\.]+)\]' % name) for (name, _) in attributes}

    def attr(line, key, cast_funct):
        """ Helper to get an attribute from a line of log.

        :param line: line of log to be treated.
        :param key: name of the attribute to lookup.
        :param cast_funct: function to use to get the appropriate type.
        :return: the value of the attribute or None if the attribute is not present.
        """
        res = re.search(patterns[key], line)
        if res is not None:
            return cast_funct(res.groups(1)[0])
        return None

    # Get logs in usable structure
    for model, idx in zip(models, range(len(models))):
        with open('../outputs/{0}/logs/{0}.log'.format(model)) as log_file:
            for line in log_file:
                iteration = attr(line, 'iteration', int)

                # get attr values
                mode = attr(line, 'mode', str)
                rand_seed = attr(line, 'rand_seed', int)

                # skip invalid lines
                if mode is None:
                    continue

                # train
                if mode == 'train':
                    train_log_loss_iteration[idx].append(iteration)
                    train_log_loss[idx].append(attr(line, 'log_loss', float))

                # test
                # same test data
                elif rand_seed == RAND_SEED:
                    test_same_iteration[idx].append(iteration)
                    for name, type in attributes:
                        test_same[idx][name].append(attr(line, name, type))

                # random test data
                else:
                    test_changing_iteration[idx].append(iteration)
                    for name, type in attributes:
                        test_changing[idx][name].append(attr(line, name, type))

    # Plot logs
    for idx in range(len(models)):
        model_mva = moving_average(train_log_loss[idx], MVA_LENGTH_TRAIN)
        plt.plot(train_log_loss_iteration[idx][MVA_LENGTH_TRAIN-1:], model_mva, '-')
    plt.legend(models, loc='upper right')
    plt.ylabel('Log loss (moving average over iterations ]x-%d, x])' % MVA_LENGTH_TRAIN)
    plt.yscale('log')
    plt.xlabel('Last iteration of the moving average')
    plt.title('Train loss evolution over training')
    plt.show()

    for idx in range(len(models)):
        test_same_mva = moving_average(test_same[idx]['log_loss'], MVA_LENGTH_TEST_SAME)
        plt.plot(test_same_iteration[idx][MVA_LENGTH_TEST_SAME-1:], test_same_mva, '-')
    plt.legend(models, loc='upper right')
    plt.ylabel('Log loss (moving average over %d last tests)' % MVA_LENGTH_TEST_SAME)
    plt.yscale('log')
    plt.xlabel('Last iteration of the moving average')
    plt.title('Test loss evolution over training (same test data)')
    plt.show()

    for idx in range(len(models)):
        test_changing_mva = moving_average(test_changing[idx]['log_loss'], MVA_LENGTH_TEST_CHANGING)
        plt.plot(test_changing_iteration[idx][MVA_LENGTH_TEST_CHANGING-1:], test_changing_mva, '-')
    plt.legend(models, loc='upper right')
    plt.ylabel('Log loss (moving average over %d last tests)' % MVA_LENGTH_TEST_CHANGING)
    plt.yscale('log')
    plt.xlabel('Last iteration of the moving average')
    plt.title('Test loss evolution over training (random test data)')
    plt.show()

    # F-measure
    for mod in ['', '_mod']:
        P_mva = [[] for _ in range(len(models))]
        R_mva = [[] for _ in range(len(models))]
        F_measure_mva = [[] for _ in range(len(models))]
        for idx in range(len(models)):
            sum_TP = moving_average(test_changing[idx]['TP' + mod], MVA_LENGTH_TEST_CHANGING, average=False)
            sum_FP = moving_average(test_changing[idx]['FP' + mod], MVA_LENGTH_TEST_CHANGING, average=False)
            sum_FN = moving_average(test_changing[idx]['FN' + mod], MVA_LENGTH_TEST_CHANGING, average=False)
            for i in range(len(sum_TP)):
                P_mva[idx].append(sum_TP[i] / (sum_TP[i] + sum_FP[i]) if sum_TP[i] > 0 else 0)
                R_mva[idx].append(sum_TP[i] / (sum_TP[i] + sum_FN[i]) if sum_TP[i] > 0 else 0)
                F_val = 2 * P_mva[idx][i] * R_mva[idx][i] / (P_mva[idx][i] + R_mva[idx][i]) if P_mva[idx][i] > 0 else 0
                F_measure_mva[idx].append(F_val)

        for idx in range(len(models)):
            plt.plot(test_changing_iteration[idx][MVA_LENGTH_TEST_CHANGING-1:], F_measure_mva[idx], '-')
        plt.legend(models, loc='lower right')
        plt.ylabel('F-measure%s (computed with %d last tests)' % (mod, MVA_LENGTH_TEST_CHANGING))
        plt.xlabel('Training iteration of the last test')
        plt.title('F-measure%s evolution over training' % mod)
        plt.show()

        for idx in range(len(models)):
            plt.plot(test_changing_iteration[idx][MVA_LENGTH_TEST_CHANGING-1:], P_mva[idx], '-')
        plt.legend(models, loc='lower right')
        plt.ylabel('Precision%s (computed with %d last tests)' % (mod, MVA_LENGTH_TEST_CHANGING))
        plt.xlabel('Training iteration of the last test')
        plt.title('Precision%s evolution over training' % mod)
        plt.show()

        for idx in range(len(models)):
            plt.plot(test_changing_iteration[idx][MVA_LENGTH_TEST_CHANGING-1:], R_mva[idx], '-')
        plt.legend(models, loc='lower right')
        plt.ylabel('Recall%s (computed with %d last tests)' % (mod, MVA_LENGTH_TEST_CHANGING))
        plt.xlabel('Training iteration of the last test')
        plt.title('Recall%s evolution over training' % mod)
        plt.show()
