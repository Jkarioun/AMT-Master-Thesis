from config import *
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':

    test_ll_kelz = []
    test_ll_mod = []

    train_ll_kelz = []
    train_ll_mod = []

    with open(PATH_LOGS + 'trace.log') as log_file:
        iter = 0

        line_iter = 0
        line_iter_test = 0

        line_num = 0
        for line in log_file:
            if 'Iteration' in line:
                iter = int(line.split(' ')[-1])

                if iter % 5 == 1:
                    line_iter = line_num
                if iter % 500 == 1:
                    line_iter_test = line_num

            # get test log losses on 4th and 5th line after the iteration line
            if line_num - line_iter == 1:
                train_ll_kelz.append(math.log(float(line.split(' ')[-1])))
            elif line_num - line_iter == 2:
                train_ll_mod.append(math.log(float(line.split(' ')[-1])))

            if line_num - line_iter_test == 4:
                test_ll_kelz.append(math.log(float(line.split(' ')[-1])))
            elif line_num - line_iter_test == 5:
                test_ll_mod.append(math.log(float(line.split(' ')[-1])))

            line_num += 1

    mv_length = 300
    train_ll_kelz = [sum(train_ll_kelz[i:i + mv_length]) / mv_length for i in range(len(train_ll_kelz) - mv_length)]
    train_ll_mod = [sum(train_ll_mod[i:i + mv_length]) / mv_length for i in range(len(train_ll_mod) - mv_length)]

    plt.plot(range(len(train_ll_kelz)), train_ll_kelz, '--', \
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
