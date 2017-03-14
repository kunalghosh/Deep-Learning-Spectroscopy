# Following example: http://seaborn.pydata.org/generated/seaborn.jointplot.html
import sys,os,pdb
import numpy as np
import pandas as pd
np.random.seed(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

vis_type = sys.argv[1]
dump_dir = sys.argv[2]

os.chdir(dump_dir)

print("Chanded dir to {}".format(dump_dir))

Y_pred = np.load("Y_test_pred_best.npz")["Y_test_pred"]
Y_test = np.load("Y_vals.npz")["Y_test"]

if vis_type == "energies":
    
    for idx in range(20):
        fig = sns.jointplot(x=Y_pred[:,idx], y=Y_test[:,idx])
        sns.plt.title("Covariate {:0>2d}".format(idx+1))
        sns.plt.xlabel("Y_pred")
        sns.plt.ylabel("Y_test")
        fig.savefig("{:0>2d}.png".format(idx+1))
elif vis_type == "spectrum":
    """
    Plots spectrum with the best and the worst MAE
    """
    mae_values = np.mean(np.abs(Y_pred-Y_test), axis=1)
    min_mae_idx = np.argmin(mae_values)
    max_mae_idx = np.argmax(mae_values)
    x_vals = np.linspace(-30,0,300)
    def plot_fig(idx, title, filename):
        fig = plt.figure()
        plt.plot(x_vals, Y_test[idx,:], label="True")
        plt.title(title)
        plt.hold(True)
        plt.plot(x_vals, Y_pred[idx,:], label="Pred")
        plt.legend()
        fig.savefig(filename)
        plt.close()
    #---- Plotting spectrum with lowest mae
    plot_fig(min_mae_idx, title="Spectrum with Lowest MAE", filename="min_mae.png")
    #---- Plotting spectrum with highest mae
    plot_fig(max_mae_idx, title="Spectrum with Highest MAE", filename="max_mae.png")
else:
    print("Supported vis_types = {}".format(["energies", "spectrum"]))
