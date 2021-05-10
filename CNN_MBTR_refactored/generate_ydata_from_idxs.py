import click
import tables
import numpy as np


@click.command()
@click.argument("spectra_path", type=click.Path(exists=True))
@click.argument("idxsfile_path", type=click.Path(exists=True))
@click.argument("yvals_path")
def generate_ydata(spectra_path, idxsfile_path, yvals_path):
    """
    Generate the Y_vals numpy file (Y_vals.npz) from the file with all Y values
    The numpy file has the test and validation splits of Y along with mean and std
    of the training split. The training split is not saved.

    Arguments
    =========
        spectra_path : str
            Path to the NumPy (.npz) file containing the spectra values (or Y, target values). 

        idxsfile_path : str
            Path to the numpy file (.npz) containing the indices of the train, validation
            and test splits of the data.

        yvals_path : str
            Path where the NumPy (.npz) file with test and validation splits of Y are
            saved along with training data's mean and standard deviation.

    Output
    ======
        NumPy file (.npz) with keys Y_test, Y_val, Y_mean, Y_std
    """

    assert spectra_path.endswith("npz"), "Expecting a NumPy (.npz) file as the first argument"
    assert idxsfile_path.endswith("npz"), "Expecting a NumPy (.npz) file as the second argument"

    y = np.load(spectra_path)["spectra"] 

    idxs = np.load(idxsfile_path) 
    idxs_train = idxs["idxs_train"]
    idxs_test = idxs["idxs_test"]
    idxs_valid = idxs["idxs_valid"]
    
    y_train = y[idxs_train, :]
    y_test = y[idxs_test, :]
    y_val = y[idxs_valid, :]

    y_train_mean = np.mean(y_train, axis=0) 
    y_train_std = np.std(y_train, axis=0)
    
    #######################################################
    ##  Useful when we want to calculate mean and std    ##
    ##  In a streaming fashion. e.g. for PyTables files. ##
    #######################################################

    # # calculate the training mean
    # for idx, count in enumerate(idxs_train):
    #     y_mean += y[idx, :]
    # y_mean /= count 

    # # calculate the standard deviation
    # for idx, count in enumerate(idxs_train):
    #     y_std += (y[idx, :] - y_mean)**2
    # y_std = np.sqrt(y_std / (count-1))

    np.savez(yvals_path, Y_train=y_train, Y_test=y_test, Y_val=y_val, Y_mean=y_train_mean, Y_std=y_train_std)

if __name__ == "__main__":
    generate_ydata()
