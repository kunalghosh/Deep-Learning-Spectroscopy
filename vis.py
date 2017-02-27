# Following example: http://seaborn.pydata.org/generated/seaborn.jointplot.html
import sys,os
import numpy as np
import pandas as pd
np.random.seed(0)
import seaborn as sns

dump_dir = sys.argv[1]
os.chdir(dump_dir)

print("Chanded dir to {}".format(dump_dir))

sns.set(style="white", color_codes=True)
Y_pred = np.loadtxt("Y_pred.txt")
Y_test = np.loadtxt("Y_test.txt")

for idx in range(20):
    fig = sns.jointplot(x=Y_pred[:,idx], y=Y_test[:,idx])
    sns.plt.title("Covariate {:0>2d}".format(idx+1))
    sns.plt.xlabel("Y_pred")
    sns.plt.ylabel("Y_test")
    fig.savefig("{:0>2d}.png".format(idx+1))
