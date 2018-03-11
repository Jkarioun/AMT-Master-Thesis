#!/usr/bin/python3.5

from config import *

if DISPLAY:
    import math
    import matplotlib.pyplot as plt
    import re

    if __name__ == '__main__':

        train_log_loss = []
        test_log_loss = []

        patterns = {
            'iteration': re.compile('\[iteration=(\d+)\]'),
            'TP': re.compile('\[TP=(\d+)\]'),
            'TN': re.compile('\[TN=(\d+)\]'),
            'FP': re.compile('\[FP=(\d+)\]'),
            'FN': re.compile('\[FN=(\d+)\]'),
            'log_loss': re.compile('\[log_loss=(.*)\]'),
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


        with open(PATH_LOGS + '%s.log' % CONFIG_NAME) as log_file:
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

                # train
                if mode == 'train':
                    train_log_loss.append(math.log(log_loss))
                # test
                else:
                    test_log_loss.append(math.log(log_loss))

                line_num += 1

        mv_length = 300
        train_ll_kelz = [sum(train_log_loss[i:i + mv_length]) / mv_length for i in
                         range(len(train_log_loss) - mv_length)]

        plt.plot(range(len(train_log_loss)), train_log_loss, '-')

        plt.legend(['kelz', 'our model'], loc='upper right')
        plt.ylabel('Log loss (moving average over 300 steps)')
        plt.xlabel('Steps (5 iterations per steps)')
        plt.title('Train loss evolution over training')

        plt.show()

        plt.plot(range(len(test_log_loss)), test_log_loss, '-')

        plt.legend(['kelz', 'our model'], loc='upper right')
        plt.ylabel('Log loss')
        plt.xlabel('Steps (500 training iterations per step)')
        plt.title('Test loss evolution over training')

        plt.show()

        # accuracy
        #        accuracy_mod = accuracy_mod[1:]
        #        accuracy_kelz = accuracy_kelz[1:]
        #        plt.plot(range(len(accuracy_kelz)), accuracy_kelz, '--',
        #                 range(len(accuracy_mod)), accuracy_mod, '-')
        #
        #        plt.legend(['kelz', 'our model'], loc='upper right')
        #        plt.ylabel('Accuracy')
        #        plt.xlabel('Steps (500 training iterations per step)')
        #        plt.title('Test accuracy evolution')

        plt.show()
