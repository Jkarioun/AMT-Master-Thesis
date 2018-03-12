import shutil
import errno
import os

PATH_SRC = '../src/'

def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def modify_config(change_dico, new_src):
    shutil.copy(PATH_SRC + 'config.py', new_src + 'config2.py')
    count_changed = 0
    with open(new_src + 'config2.py', 'r') as f2, open(new_src + 'config.py', 'w') as f1:
        for line in f2:
            changed = False
            for entry in change_dico:
                if line.startswith(entry):
                    assert not changed, 'two times changed'
                    f1.write(entry+change_dico[entry]+'\n')
                    changed = True
                    count_changed += 1
            if not changed:
                f1.write(line)
    assert count_changed==len(change_dico)
    os.remove(new_src + 'config2.py')

for i in range(73):
    new_src = '../srcc' + str(i) + '/'
    dico = {}
    dico['PATH_SRC = '] = '\'' + new_src + '\''
    if i == 0:
        dico['CONFIG_NAME = '] = '\'frame_weighted_hybrid_LR0002_UP0\''
    elif i == 8:
        dico['CONFIG_NAME = '] = '\'frame_weighted_hybrid_LR001_UP0\''
        dico['LEARNING_RATE = '] = '0.001'
    elif i == 16:
        dico['CONFIG_NAME = '] = '\'frame_weighted_hybrid_LR00005_UP0\''
        dico['LEARNING_RATE = '] = '0.00005'
    elif i == 24:
        dico['CONFIG_NAME = '] = '\'frame_weighted_kelz_LR0002_UP0\''
        dico['KELZ_MODEL = '] = 'True'
    elif i == 32:
        dico['CONFIG_NAME = '] = '\'frame_weighted_kelz35_LR0002_UP0\''
        dico['KELZ_MODEL = '] = 'True'
        dico['KELZ_KERNEL = '] = '[3, 5]'
    elif i == 40:
        dico['CONFIG_NAME = '] = '\'frame_weighted_hybrid37_LR0002_UP0\''
        dico['HARMONIC_RELATIVES = '] = 'np.array([1 / 5, 1 / 3, 1 / 2, 1, 2, 3, 5])'
    elif i == 48:
        dico['CONFIG_NAME = '] = '\'frame_weighted_harm_LR0002_UP0\''
        dico['FIRST_LAYER_HARMONIC = '] = 'True'
    elif i == 56:
        dico['CONFIG_NAME = '] = '\'onset_weighted_hybrid_LR0002_UP0\''
        dico['ONSET = '] = 'True'
    elif i == 64:
        dico['CONFIG_NAME = '] = '\'onset_weighted_kelz_LR0002_UP0\''
        dico['ONSET = '] = 'True'
        dico['KELZ_MODEL = '] = 'True'
    elif i == 72:
        dico['CONFIG_NAME = '] = '\'onset_weighted_harm_LR0002_UP0\''
        dico['ONSET = '] = 'True'
        dico['FIRST_LAYER_HARMONIC = '] = 'True'
    else:
        dico['TESTING = '] = 'False'
        dico['TRAINING = '] = 'False'
        dico['CONFIG_NAME = '] = '\'dummy\''
        dico['CREATE_OUTPUT_FOLDERS = '] = 'False'
    copy(PATH_SRC, new_src)
    modify_config(dico, new_src)
