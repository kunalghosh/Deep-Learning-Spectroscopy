"""
Here we implement a baseline simple neural network for the 7K_20_eigenvalues
dataset. It consists of two parts: the coulomb.txt and energies.txt with the
corresponding data.

The coulomb.txt has a set of 7089 matrices each of dimensions 29x29
The energies.txt is a matrix of 7089 rows and 20 columns.
"""

import numpy as np
import pdb

seed = np.random.randint(4294967295)
print("seed {}".format(seed))
np.random.seed(seed)

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import *

def load_data(coulomb_txtfile, energies_txtfile):
    Y = np.loadtxt(energies_txtfile).astype(np.float32) # Note that here y_i is a vector of 20 values
    X = np.loadtxt(coulomb_txtfile).reshape((-1,29,29,1)).astype(np.float32)
    return X,Y

def preprocess_targets(Y):
    """
    zero mean and unit variance
    """
    _mean = Y.mean(axis=0)
    _std  = Y.std(axis=0)
    Y = Y - _mean
    Y = Y / _std
    return Y, _mean, _std

def get_data_splits(X,Y,splits=None, randomize=None):
    """
    * X and Y must be numpy "arrays"
    * Returns the training and test splits of the data. 
    * If splits are not defined then default splits of 75 and 25 percent is used.
    * You can pass a list splits = [train, validation, test] each of the values is
    an integer and the train + validation + test must be equal to 100.
    * You can also pass a list splits = [train, test] where train and test are integers
    and train + test must be equal to 100.
    """
    if splits is None:
        splits = [75, 25]
    else:
        assert len(splits) in [2,3], "splits must be a list of length 2 or 3"
        assert sum(splits) is 100
    num_datapoints = Y.shape[0]
    indices = range(num_datapoints)
    if randomize is not None:
        indices = np.random.permutation(indices)

    # The splits are rounded to the nearest integer
    index_splits = np.floor(np.cumsum(splits) * num_datapoints / 100).astype(np.int32)
    # The last split is empty when the count of splits adds up to 100%
    return np.split(X, index_splits)[:-1], np.split(Y, index_splits)[:-1], index_splits

def get_model():
    """
    Here we implement the Neural Network model
    """
    input_conv_layer = Convolution2D(5,3,3, activation='relu', border_mode='same', input_shape=(29,29,1))

    model_real = Sequential([
        input_conv_layer,
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        MaxPooling2D((2,2)),
        # Feature map is (14,14) now
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        MaxPooling2D((2,2)),
        # Feature map is (7,7) now
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        # Feature map shape is [atch, height, width, depth=5]
        Flatten(),
        # Feature map shape is [batch, height*width*depth] = [batch, 245]
        Dense(20, activation='linear')
        ])

    model_binary = Sequential([
        input_conv_layer,
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        MaxPooling2D((2,2)),
        # Feature map is (14,14) now
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        MaxPooling2D((2,2)),
        # Feature map is (7,7) now
        Convolution2D(5,3,3, activation='relu', border_mode='same'),
        # Feature map shape is [atch, height, width, depth=5]
        Flatten(),
        # Feature map shape is [batch, height*width*depth] = [batch, 245]
        Dense(20, activation='hard_sigmoid')
        ])

    merged = Merge([model_real, model_binary], mode='mul')
    model = Sequential([
        merged,
        Dense(20, activation='linear')
        ])

    model.compile(loss='mae', optimizer='adam', metrics=['accuracy','mean_absolute_error'])
    return model

if __name__ == "__main__":
    coulomb_path  = sys.argv[1]
    energies_path = sys.argv[2]
    X,Y = load_data("/m/home/home0/00/ghoshk1/data/Desktop/Thesis/data/7k_20_eigenvalues/coulomb.txt",
                    "/m/home/home0/00/ghoshk1/data/Desktop/Thesis/data/7k_20_eigenvalues/energies.txt")
    Y, Y_mean, Y_std = preprocess_targets(Y)
    [X_train, X_test], [Y_train, Y_test], splits = get_data_splits(X,Y, splits=[90,10])

    model = get_model()
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1, save_best_only=True)
    model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            batch_size = 10,
            nb_epoch = 800,
            verbose = 1,
            shuffle = True,
            callbacks=[tensorboard_callback, checkpointer]
            )
    # evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save_weights("model.h5")
    print("Saved model weights to disk")

    Y_pred = model.predict(X_test)
    print("Got predictions: saved in Y_pred")
    pdb.set_trace()
