from config import *
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':

    test_ll_kelz = []
    test_ll_mod = []
    with open(PATH_LOGS + 'trace.log') as log_file:
        iter = 0
        line_iter = 0
        line_num = 0
        for line in log_file:
            if 'Iteration' in line:
                iter = int(line.split(' ')[-1])

                if iter % 500 == 1:
                    line_iter = line_num

            # get test log losses on 4th and 5th line after the iteration line
            if line_num - line_iter == 4:
                test_ll_kelz.append(math.log(float(line.split(' ')[-1])))
            elif line_num - line_iter == 5:
                test_ll_mod.append(math.log(float(line.split(' ')[-1])))

            line_num += 1

    plt.plot(range(len(test_ll_kelz)), test_ll_kelz, '--', \
              range(len(test_ll_kelz)),test_ll_mod, '-')

    plt.legend(['kelz','mod'], loc='upper right')

    plt.show()
