from config import *
import math
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':

    test_ll_kelz = []
    test_ll_mod = []

    train_ll_kelz = []
    train_ll_mod = []

    patterns = {
        'iteration': re.compile('\[iteration=(\d+)\]'),
        'measure': re.compile('\[measure=(\w+)\]'),
        'onset': re.compile('\[onset=(\w+)\]'),
        'model': re.compile('\[model=(\w+)\]')
    }


    def iteration(line):
        res = re.search(patterns['iteration'], line)
        if res is not None:
            return int(res.group(1))
        return None


    def attr(line, key):
        res = re.search(patterns[key], line)
        if res is not None:
            return res.groups(1)[0]
        return None


    with open(PATH_LOGS + '%s.log' % CONFIG_NAME) as log_file:
        iter = 0

        line_iter = 0
        line_iter_test = 0

        line_num = 0
        last_line_kelz_test = False
        for line in log_file:
            iter = iteration(line)

            # train
            if iter is not None:
                if attr(line, 'model') == 'kelz' and attr(line, 'measure') == 'log_loss':
                    train_ll_kelz.append(math.log(float(line.split(' ')[-1])))
                elif attr(line, 'model') == 'mod' and attr(line, 'measure') == 'log_loss':
                    train_ll_mod.append(math.log(float(line.split(' ')[-1])))
            # test
            else:
                if attr(line, 'model') == 'mod' and attr(line, 'measure') == 'log_loss':
                    # temporary bugfix (mod is output twice in logs)
                    if last_line_kelz_test:
                        test_ll_mod.append(math.log(float(line.split(' ')[-1])))
                        last_line_kelz_test = False
                    else:
                        test_ll_kelz.append(math.log(float(line.split(' ')[-1])))
                        last_line_kelz_test = True
                # elif attr(line, 'model') == 'kelz' and attr(line, 'measure') == 'log_loss':


            line_num += 1

    mv_length = 300
    train_ll_kelz = [sum(train_ll_kelz[i:i + mv_length]) / mv_length for i in range(len(train_ll_kelz) - mv_length)]
    train_ll_mod = [sum(train_ll_mod[i:i + mv_length]) / mv_length for i in range(len(train_ll_mod) - mv_length)]

    if DISPLAY:
        plt.plot(range(len(train_ll_kelz)), train_ll_kelz, '--',
             range(len(train_ll_kelz)), train_ll_mod, '-')

        plt.legend(['kelz', 'our model'], loc='upper right')
        plt.ylabel('Log loss (moving average over 300 steps)')
        plt.xlabel('Steps (5 iterations per steps)')
        plt.title('Train loss evolution over training')

        plt.show()

        plt.plot(range(len(test_ll_kelz)), test_ll_kelz, '--', \
                 range(len(test_ll_kelz)), test_ll_mod, '-')

        plt.legend(['kelz', 'our model'], loc='upper right')
        plt.ylabel('Log loss')
        plt.xlabel('Steps (500 training iterations per step)')
        plt.title('Test loss evolution over training')

        plt.show()
    else:
        print("DISPLAY is set to False")
