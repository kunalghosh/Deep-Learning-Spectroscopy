import sys
import pdb
import numpy as np

def main(filename):
    """
    Loads a file with coulomb matrices and tries to find its shape
    from existing shapes (-1, 23,23) or (-1, 29,29)
    """
    coulomb_mats = np.loadtxt(filename)
    try:
        coulomb_mats = coulomb_mats.reshape(-1, 23,23)
    except:
        print("Not 23x23")
        try:
            coulomb_mats = coulomb_mats.reshape(-1, 29,29)
        except:
            print("Not 29x29\n [Error] Don't know shape of coulomb matrix. Dropping into debugger.")
            pdb.set_trace()

    return  coulomb_mats


if __name__ == "__main__":
    filename = sys.argv[1]
    coulomb_mats = main(filename)
    print("coulomb shape = {}".format(coulomb_mats.shape))
