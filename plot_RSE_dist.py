import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


data = np.load("Y_test_pred_best_mae.npz")
predictions = data["Y_test_pred"]

y_vals = np.load("Y_vals.npz")["Y_test"]

rse_vals = []


def relative_difference(prediction, label):
    dE = 30/len(prediction) #how many eV's one dE is
    numerator = np.sum(np.power(dE*(label-prediction),2))
    denominator = np.sum(dE*label)
    return np.sqrt(numerator)/denominator


for idx, (pred, label) in enumerate(zip(predictions, y_vals)):
    rse_vals.append(relative_difference(pred, label))

plt.hist(rse_vals, bins=50)
plt.savefig("dist_mae.pdf")
