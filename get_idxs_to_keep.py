import numpy as np

def get_idxs_to_keep(path_to_energies_file):
    energies = np.loadtxt(path_to_energies_file)
    idxs = (energies[:,15] != -10000.000)
    return idxs
