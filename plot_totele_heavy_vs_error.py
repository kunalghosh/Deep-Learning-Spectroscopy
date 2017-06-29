import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


import re
def sort_nicely( l ):
    """
    Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    

totele=[]
heavy=[]
files = glob.glob("mse_testset_prediction*.png")
sort_nicely(files)
for f in files:#os.listdir("."):
    #if f.startswith("mse_testset_prediction"):
    print(f.split("-")[0])
    te, he = map(int,f.split("-")[1:3])
    totele.append(te)
    heavy.append(he)

fig = plt.figure()
xvals = range(len(totele))
plt.scatter(xvals, totele, label="Total atoms.")
plt.hold(True)
plt.scatter(xvals, heavy, label="Heavy atoms")
plt.grid(True)
plt.legend()
plt.title("Total atoms, Heavy elements as a function of erro (left low error, right high error)")
plt.xlabel("Error (low to high)")
plt.ylabel("Count")
plt.savefig("totele_heavy_vs_error.png")
plt.close()


        
