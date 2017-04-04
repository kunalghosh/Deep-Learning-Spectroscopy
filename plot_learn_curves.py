import re
import numpy as np
# given the path to a log file log_file.log and a data.txt file
# script returns the train_value, test_value, train_legend, test_legend, x_label, y_label

def get_learn_curve_data(logfile, datafile):

    # getting the learning curve values
    train_vals = []
    test_vals = []

    rexp_train = re.compile(r'Train .* \d*.\d*$')
    rexp_test = re.compile(r'Test .* \d*.\d*$')

    with open(logfile) as f:
        for line in f:
            match = rexp_train.search(line)
            if match == None:
                match = rexp_test.search(line)
            if match == None:
                pass
            else:
                split = match.group().split()
                if split[0] == "Test":
                    test_vals.append(np.float32(split[2]))
                elif split[0] == "Train":
                    train_vals.append(np.float32(split[2]))
                else:
                    print("Neither test nor train !! --> {}".format(line))

    # getting the frequency of values
    plotevery = 10 # default value used in scripts
    with open(datafile) as f:
        for line in f:
            match = re.search(r'plotevery .*', line)
            if match:
                plotevery = np.int32(match.group().split()[2])
                break

    test_legend = "Test"
    train_legend = "Train"
    x_label = "Epochs (at intervals of %d)" % plotevery
    y_label = "Loss"

    return train_vals, test_vals, train_legend, test_legend, x_label, y_label

