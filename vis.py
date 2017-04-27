# Following example: http://seaborn.pydata.org/generated/seaborn.jointplot.html
import sys,os,pdb
import numpy as np
import pandas as pd
np.random.seed(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
sns.set(style="white", color_codes=True)
from plot_learn_curves import get_learn_curve_data

vis_type = sys.argv[1]
dump_dir = sys.argv[2]

os.chdir(dump_dir)

def plot_fig(idx, title, filename):
    fig = plt.figure()
    plt.plot(x_vals, Y_test[idx,:], label="True")
    plt.title(title)
    plt.hold(True)
    plt.plot(x_vals, Y_pred[idx,:], label="Pred")
    plt.legend()
    plt.xlabel("Energies [eV]")
    plt.ylabel("Intensity")
    fig.savefig(filename)
    plt.close()

def plothist(data, title, filename):
    fig = plt.figure()
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel("Error")
    fig.savefig(filename)
    plt.close()
    
def Rsq(X,Y):
    Rsq.__name__ = r"$R^2$"
    return pearsonr(X,Y)[0]**2

print("Chanded dir to {}".format(dump_dir))

Y_pred = np.load("Y_test_pred_best.npz")["Y_test_pred"]
Y_test = np.load("Y_vals.npz")["Y_test"]
Y_mean = np.load("Y_vals.npz")["Y_mean"]
Y_std = np.load("Y_vals.npz")["Y_std"]
Y_pred = Y_pred * Y_std + Y_mean
Y_test = Y_test * Y_std + Y_mean

if vis_type == "energies":

    _, num_energies = Y_pred.shape
    for idx in range(num_energies):
        
        _min = min([min(Y_pred[:,idx]),min(Y_test[:,idx])])
        _max = max([max(Y_pred[:,idx]),max(Y_test[:,idx])])

        plt.xlim([_min, _max])
        plt.ylim([_min, _max])

        fig = sns.jointplot(x=Y_pred[:,idx], y=Y_test[:,idx], stat_func=Rsq)

        # x0, x1 = fig.ax_joint.get_xlim()
        # y0, y1 = fig.ax_joint.get_ylim()
        
    
        fig.ax_joint.plot([_min,_max],[_min,_max],":k")
        sns.plt.title("Eigenvalue {:0>2d}".format(idx+1))
        # sns.plt.xlabel("Y_pred [eV]")
        # sns.plt.ylabel("Y_test [eV]")
        fig.savefig("{:0>2d}_testset_prediction.png".format(idx+1))
        plt.close()
elif vis_type == "spectrum_mse":
    """
    Plots spectrum with the best and the worst MSE
    """
    mse_values = np.mean((Y_pred-Y_test)**2, axis=1)
    # -- Plotting histogram
    plothist(mse_values,"Histogram of MSE (test set)","hist_mse.png") 
    # -- end plotting histogram
    min_mse_idx = np.argmin(mse_values)
    max_mse_idx = np.argmax(mse_values)
    x_vals = np.linspace(-30,0,300)
    #---- Plotting spectrum with lowest mse
    plot_fig(min_mse_idx, title="Spectrum with Lowest MSE (= {:>0.4})".format(mse_values[min_mse_idx]), filename="min_mse_testset_prediction.png")
    #---- Plotting spectrum with highest mse
    plot_fig(max_mse_idx, title="Spectrum with Highest MSE (= {:>0.4})".format(mse_values[max_mse_idx]), filename="max_mse_testset_prediction.png")

elif vis_type == "spectrum_mae":
    """
    Plots spectrum with the best and the worst MAE
    """
    mae_values = np.mean(np.abs(Y_pred-Y_test), axis=1)
    plothist(mae_values,"Histogram of MAE (test set)","hist_mae.png") 
    min_mae_idx = np.argmin(mae_values)
    max_mae_idx = np.argmax(mae_values)
    x_vals = np.linspace(-30,0,300)
    #---- Plotting spectrum with lowest mae
    plot_fig(min_mae_idx, title="Spectrum with Lowest MAE (= {:>0.4})".format(mae_values[min_mae_idx]), filename="min_mae.png")
    #---- Plotting spectrum with highest mae
    plot_fig(max_mae_idx, title="Spectrum with Highest MAE (= {:>0.4})".format(mae_values[max_mae_idx]), filename="max_mae.png")
else:
    print("Supported vis_types = {}".format(["energies", "spectrum_mae", "spectrum_mse"]))

# saving the learning curve 
print("Trying to generate the learning curve now.")
try:
    train_vals, test_vals, train_legend, test_legend, x_label, y_label = get_learn_curve_data("log_file.log", "data.txt") 
    fig = plt.figure()
    x = range(len(train_vals))
    plt.plot(x, test_vals, label=test_legend)
    plt.hold(True)
    plt.plot(x, train_vals, label=train_legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.title("Learning curves")
    plt.savefig("learning_curves.png")
    plt.close()
except Exception as e:
    print("Couldn't generate learning curves : {}".format(e))
