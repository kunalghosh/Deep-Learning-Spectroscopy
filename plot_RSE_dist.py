import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def get_hist(name="rmse"):#alternately name could be "mse"
    data = np.load(f"Y_test_pred_best_{name}.npz")
    predictions = data["Y_test_pred"]

    y_vals = np.load("Y_vals.npz")["Y_test"]

    def relative_difference(prediction, label):
        dE = 30/len(prediction) #how many eV's one dE is
        numerator = np.sum(dE * np.power((label-prediction),2))
        denominator = np.sum(dE*label)
        return np.sqrt(numerator)/denominator

    rse_vals = []
    for idx, (pred, label) in enumerate(zip(predictions, y_vals)):
        rse_vals.append(relative_difference(pred, label))

    # plt.xticks(np.arange(0, 0.16, step=0.3))
    plt.hist(rse_vals, bins=50)
    plt.axvline(x=0.03, color='k')
    plt.savefig(f"dist_{name}.pdf")
    plt.close()


get_hist("rmse")
get_hist("mae")
