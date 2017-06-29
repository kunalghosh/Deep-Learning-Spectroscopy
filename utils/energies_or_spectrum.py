import sys
import pdb
import numpy as np

def main(filename):
    """
    Loads a file with energies or spectrum 
    """
    coulomb_mats = np.loadtxt(filename)
    return  coulomb_mats


if __name__ == "__main__":
    filename = sys.argv[1]
    energies_or_spectrum = main(filename)
    print("shape = {}".format(energies_or_spectrum.shape))
