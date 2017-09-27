"""
Here we implement a baseline simple neural network for the 7K_20_eigenvalues
dataset. It consists of two parts: the coulomb.txt and energies.txt with the
corresponding data.

The coulomb.txt has a set of 7089 matrices each of dimensions 29x29
The energies.txt is a matrix of 7089 rows and 20 columns.
"""
import os,sys
import numpy as np
import pdb

seed = np.random.randint(4294967295)
print("seed {}".format(seed))
np.random.seed(seed)

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.objectives import binary_crossentropy, mae
from keras.layers import *

def load_data(coulomb_txtfile, energies_txtfile):
    Y = np.loadtxt(energies_txtfile).astype(np.float32) # Note that here y_i is a vector of 20 values
    X = np.loadtxt(coulomb_txtfile).reshape((-1,29,29,1)).astype(np.float32)
    return X,Y

def preprocess_targets(Y):
    """
    zero mean and unit variance
    """
    Y_binarized = (Y != 0).astype(np.float32)
    _mean = Y.mean(axis=0)
    _std  = Y.std(axis=0)
    Y = Y - _mean
    Y = Y / _std
    return Y, _mean, _std, Y_binarized

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
    input_coulomb = Input(shape=(29,29,1))
    input_eigs = Input(shape=(20,1))
    input_eigs_binary = Input(shape=(20,1))

    m1_conv_layer1 = Convolution2D(5,3,3, activation='relu', border_mode='same')(input_coulomb)
    m1_conv_layer2 = Convolution2D(5,3,3, activation='relu', border_mode='same')(m1_conv_layer1)
    m1_maxpool_1 = MaxPooling2D((2,2))(m1_conv_layer2)
    # Feature map is (14,14) now
    m1_conv_layer3 = Convolution2D(5,3,3, activation='relu', border_mode='same')(m1_maxpool_1)
    m1_maxpool_2 = MaxPooling2D((2,2))(m1_conv_layer3)
    # Feature map is (7,7) now
    m1_conv_layer4 = Convolution2D(5,3,3, activation='relu', border_mode='same')(m1_maxpool_2)
    # Feature map shape is [atch, height, width, depth=5]
    m1_flattened = Flatten()(m1_conv_layer4)
    # Feature map shape is [batch, height*width*depth] = [batch, 245]
    m1_out = Dense(20, activation='linear')(m1_flattened)

    m2_conv_layer1 = Convolution2D(5,3,3, activation='relu', border_mode='same')(input_coulomb)
    m2_conv_layer2 = Convolution2D(5,3,3, activation='relu', border_mode='same')(m2_conv_layer1)
    m2_maxpool_1 = MaxPooling2D((2,2))(m2_conv_layer2)
    # Feature map is (14,14) now
    m2_conv_layer3 = Convolution2D(5,3,3, activation='relu', border_mode='same')(m2_maxpool_1)
    m2_maxpool_2 = MaxPooling2D((2,2))(m2_conv_layer3)
    # Feature map is (7,7) now
    m2_conv_layer4 = Convolution2D(5,3,3, activation='relu', border_mode='same')(m2_maxpool_2)
    # Feature map shape is [atch, height, width, depth=5]
    m2_flattened = Flatten()(m2_conv_layer4)
    # Feature map shape is [batch, height*width*depth] = [batch, 245]
    m2_out = Dense(20, activation='hard_sigmoid')(m2_flattened)

    merged = m2_out * m1_out
    merged_out = Dense(20, activation='linear')(merged)

    def new_loss(input_eigs, merged_out, m2_out):
        mae_loss = mae(input_eigs, merged_out)
        cross_ent_loss = binary_crossentropy(m2_out, input_eigs_binary) 

    model = Model(input=[input_coulomb, input_eigs, input_eigs_binary], output=merged_out)
    model.compile(loss=new_loss, optimizer='adam', metrics=['accuracy','mean_absolute_error'])
    return model

if __name__ == "__main__":
    
    data_path=sys.argv[1]

    X,Y = load_data(data_path + os.sep + "coulomb.txt",
                    data_path + os.sep + "energies.txt")
    Y, Y_mean, Y_std, Y_binarized = preprocess_targets(Y)
    [X_train, X_test], [Y_train, Y_test], splits = get_data_splits(X,Y, splits=[90,10])
    [Y_binarized_train, Y_binarized_test] = np.split(Y_binarized,splits)[:-1]
    
    model = get_model()
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1, save_best_only=True)
    model.fit(
            X_train, Y_train, Y_binarized_train,
            validation_data=(X_test, Y_test, Y_binarized_test),
            batch_size = 10,
            nb_epoch = 800,
            verbose = 1,
            shuffle = True,
            callbacks=[tensorboard_callback, checkpointer]
            )
    # evaluate the model
    scores = model.evaluate(X_test, Y_test, Y_binarized_test, verbose=0)
    print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save_weights("model.h5")
    print("Saved model weights to disk")

    Y_pred = model.predict(X_test)
    print("Got predictions: saved in Y_pred")

    np.savetxt("Y_pred.txt",Y_pred)
    np.savetxt("Y_test.txt",Y_test)
    print("Saved Y_pred and Y_test to disk")
    pdb.set_trace()
