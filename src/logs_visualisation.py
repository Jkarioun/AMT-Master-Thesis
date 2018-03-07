from config import *
import math
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':

    test_ll_kelz = []
    test_ll_mod = []

    train_ll_kelz = []
    train_ll_mod = []

    accuracy_mod = []
    accuracy_kelz = []

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
                measure = float(line.split(' ')[-1])
                if attr(line, 'model') == 'kelz' and attr(line, 'measure') == 'log_loss':
                    train_ll_kelz.append(math.log(measure))
                elif attr(line, 'model') == 'mod' and attr(line, 'measure') == 'log_loss':
                    train_ll_mod.append(math.log(measure))
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
                if attr(line, 'model') == 'mod' and attr(line, 'measure') == 'accuracy':
                    measure = float(line.split(' ')[-1])
                    accuracy_mod.append(measure)
                elif attr(line, 'model') == 'kelz' and attr(line, 'measure') == 'accuracy':
                    measure = float(line.split(' ')[-1])
                    accuracy_kelz.append(measure)


            line_num += 1

    mv_length = 300
    train_ll_kelz = [sum(train_ll_kelz[i:i + mv_length]) / mv_length for i in range(len(train_ll_kelz) - mv_length)]
    train_ll_mod = [sum(train_ll_mod[i:i + mv_length]) / mv_length for i in range(len(train_ll_mod) - mv_length)]

    plt.plot(range(len(train_ll_kelz)), train_ll_kelz, '--',
         range(len(train_ll_kelz)), train_ll_mod, '-')

    plt.legend(['kelz', 'our model'], loc='upper right')
    plt.ylabel('Log loss (moving average over 300 steps)')
    plt.xlabel('Steps (5 iterations per steps)')
    plt.title('Train loss evolution over training')

    plt.show()

    plt.plot(range(len(test_ll_kelz)-1), test_ll_kelz[1:], '--',
             range(len(test_ll_kelz)-1), test_ll_mod, '-')

    plt.legend(['kelz', 'our model'], loc='upper right')
    plt.ylabel('Log loss')
    plt.xlabel('Steps (500 training iterations per step)')
    plt.title('Test loss evolution over training')

    plt.show()

    # accuracy
    accuracy_mod = accuracy_mod[1:]
    accuracy_kelz = accuracy_kelz[1:]
    plt.plot(range(len(accuracy_kelz)), accuracy_kelz, '--',
             range(len(accuracy_mod)), accuracy_mod, '-')

    plt.legend(['kelz', 'our model'], loc='upper right')
    plt.ylabel('Accuracy')
    plt.xlabel('Steps (500 training iterations per step)')
    plt.title('Test accuracy evolution')

    plt.show()

