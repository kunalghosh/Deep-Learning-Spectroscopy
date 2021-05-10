# Following example: http://seaborn.pydata.org/generated/seaborn.jointplot.html
from __future__ import print_function
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
sns.set_palette("Dark2")
from plot_learn_curves import get_learn_curve_data

vis_type = sys.argv[1]
needs_shift_scale = sys.argv[2] # "no_shift_scale" to disable
dump_dir = sys.argv[3]

os.chdir(dump_dir)

# fontsize = 40

def plot_fig(idx, title, filename):
    # plt.rcParams.update({'font.size': fontsize})
    sns.set(style="white", color_codes=True,font_scale=1.5, rc={"lines.linewidth" : 5})
    sns.set_palette("Dark2")
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
    # plt.rcParams.update({'font.size': fontsize})
    sns.set(style="white", color_codes=True,font_scale=1.5)
    sns.set_palette("gray")

    fig = plt.figure()
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel("Error")
    fig.savefig(filename)
    plt.close()

    # sns.set(style="white", color_codes=True)
    sns.set_palette("Dark2")
    
def Rsq(X,Y):
    Rsq.__name__ = r"$R^2$"
    return pearsonr(X,Y)[0]**2

def get_printable_data(charges):
    charges = charges[np.nonzero(charges)]
    charges.sort()
    charges = charges[::-1]
    tot_ele = len(charges)
    heavy = sum(charges != 1)
    return "_".join(map(str,charges.tolist())), tot_ele, heavy, 

print("Chanded dir to {}".format(dump_dir))

Y_pred = np.load("Y_test_pred_best.npz")["Y_test_pred"]
Y_test = np.load("Y_vals.npz")["Y_test"]
Y_mean = np.load("Y_vals.npz")["Y_mean"]
Y_std = np.load("Y_vals.npz")["Y_std"]
# if needs_shift_scale == "no_shift_scale":
if needs_shift_scale == "no_scale_shift":
    Y_mean = np.zeros_like(Y_mean)
    Y_std  = np.ones_like(Y_std)
    print("No, shift and scale applied")
Y_pred = Y_pred * Y_std + Y_mean
Y_test = Y_test * Y_std + Y_mean

Z_test = None
Z_string = ""
try:
    Z_test = np.load('X_vals.npz')['Z_test']
except (KeyError, FileNotFoundError) as e:
    pass

def save_to_file(error_list, sorted_idxs, filename):
    with open(filename, "w+") as f:
        for p,si in enumerate(sorted_idxs):
            print("{} , {}".format(error_list[si],p), file=f)
    print("Saved errrors and indices to {}".format(filename))


if vis_type == "energies":

    _, num_energies = Y_pred.shape
    for idx in range(num_energies):
        
        _min = min([min(Y_pred[:,idx]),min(Y_test[:,idx])])
        _max = max([max(Y_pred[:,idx]),max(Y_test[:,idx])])

        plt.xlim([_min, _max])
        plt.ylim([_min, _max])

        # fig, ax = plt.subplots(figsize=(1.685, 1.602))
        # fig.set_size_inches(1.685, 1.602)
        # sns.jointplot(x=Y_pred[:,idx], y=Y_test[:,idx], stat_func=Rsq)
        sns.set(style="white", color_codes=True,font_scale=2.5)
        sns.set_palette("gray")
        # print(sns.axes_style())
        fig = sns.jointplot(x=Y_pred[:,idx], y=Y_test[:,idx], stat_func=Rsq).set_axis_labels("Predicted [eV]", "True [eV]")

        # x0, x1 = fig.ax_joint.get_xlim()
        # y0, y1 = fig.ax_joint.get_ylim()
        
    
        # fig.ax_joint.plot([_min,_max],[_min,_max],":k")
        fig.ax_joint.plot([_min,_max],[_min,_max],":w")
        #sns.plt.title("Eigenvalue {:0>2d}".format(idx+1))
        # sns.plt.xlabel("Predicted [eV]")
        # sns.plt.ylabel("True [eV]")
        fig.savefig("{:0>2d}_testset_prediction.png".format(idx+1))
        plt.close()
elif vis_type in ("spectrum_mse", "spectrum_mse_all"):
    """
    Plots spectrum with the best and the worst MSE
    """
    mse_values = np.sqrt(np.mean((Y_pred-Y_test)**2, axis=1))
    # -- Plotting histogram
    plothist(mse_values,"Histogram of MSE (test set)","hist_mse.png") 
    x_vals = np.linspace(-30,0,300)
    # -- end plotting histogram
    if vis_type == "spectrum_mse":
        min_mse_idx = np.argmin(mse_values)
        max_mse_idx = np.argmax(mse_values)
        #---- Plotting spectrum with lowest mse
        plot_fig(min_mse_idx, title="Spectrum with Lowest RMSE (= {:>0.4})".format(mse_values[min_mse_idx]), filename="min_mse_testset_prediction.png")
        #---- Plotting spectrum with highest mse
        plot_fig(max_mse_idx, title="Spectrum with Highest RMSE (= {:>0.4})".format(mse_values[max_mse_idx]), filename="max_mse_testset_prediction.png")
    elif vis_type == "spectrum_mse_all":
        sorted_idxs = np.argsort(mse_values)
        zeros_to_pad = np.ceil(np.log10(len(mse_values))).astype(np.int32)

        save_to_file(mse_values, sorted_idxs, filename="mse_error_filename.txt") 
        for position, sorted_idx in enumerate(sorted_idxs):
            if Z_test is not None:
                Z_string,tot_ele,heavy = get_printable_data(Z_test[sorted_idx])
                plot_fig(sorted_idx, title="Spectrum with MSE (= {:>0.4})".format(mse_values[sorted_idx]), filename=("mse_testset_prediction_{:0>%d}-{}-{}-{}" % zeros_to_pad).format(position, tot_ele, heavy, Z_string))
            else:
                plot_fig(sorted_idx, title="Spectrum with MSE (= {:>0.4})".format(mse_values[sorted_idx]), filename=("mse_testset_prediction_{:0>%d}" % zeros_to_pad).format(position))

elif vis_type in ("spectrum_mae", "spectrum_mae_all"):
    """
    Plots spectrum with the best and the worst MAE
    """
    mae_values = np.mean(np.abs(Y_pred-Y_test), axis=1)
    plothist(mae_values,"Histogram of MAE (test set)","hist_mae.png") 
    x_vals = np.linspace(-30,0,300)
    if vis_type == "spectrum_mae":
        min_mae_idx = np.argmin(mae_values)
        max_mae_idx = np.argmax(mae_values)
        #---- Plotting spectrum with lowest mae
        plot_fig(min_mae_idx, title="Spectrum with Lowest MAE (= {:>0.4})".format(mae_values[min_mae_idx]), filename="min_mae_testset_prediction.png")
        #---- Plotting spectrum with highest mae
        plot_fig(max_mae_idx, title="Spectrum with Highest MAE (= {:>0.4})".format(mae_values[max_mae_idx]), filename="max_mae_testset_prediction.png")
    elif vis_type == "spectrum_mae_all":
        sorted_idxs = np.argsort(mae_values)
        zeros_to_pad = np.ceil(np.log10(len(mae_values))).astype(np.int32)

        save_to_file(mse_values, sorted_idxs, filename="mae_error_filename.txt") 
        for position, sorted_idx in enumerate(sorted_idxs):
            if Z_test is not None:
                Z_string,tot_ele,heavy = get_printable_data(Z_test[sorted_idx])
                plot_fig(sorted_idx, title="Spectrum with MAE (= {:>0.4})".format(mae_values[sorted_idx]), filename=("mae_testset_prediction_{:0>%d}-{}-{}-{}" % zeros_to_pad).format(position,tot_ele, heavy, Z_string))
            else:
                plot_fig(sorted_idx, title="Spectrum with MAE (= {:>0.4})".format(mae_values[sorted_idx]), filename=("mae_testset_prediction_{:0>%d}" % zeros_to_pad).format(position))
else:
    print("Supported vis_types = {}".format(["energies", "spectrum_mae", "spectrum_mse", "spectrum_mae_all", "spectrum_mse_all"]))

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
