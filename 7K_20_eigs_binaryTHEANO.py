
# coding: utf-8

# In[3]:

from __future__ import print_function

import os
import sys
from datetime import datetime
import numpy as np
import pdb

import click

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
try:
    get_ipython().magic(u'matplotlib inline')
except Exception as e:
    pass

import theano 
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import lasagne
from lasagne.layers import * #InputLayer, DenseLayer, get_output, get_all_params, MergeLayer, get_all_param_values
from lasagne.nonlinearities import rectify, softmax, leaky_rectify, linear, sigmoid
from lasagne.objectives import squared_error, binary_crossentropy
from lasagne.updates import adam, sgd, momentum, nesterov_momentum

from util import get_logger


def load_data(coulomb_txtfile, energies_txtfile):
    Y = np.loadtxt(energies_txtfile).astype(np.float32) # Note that here y_i is a vector of 20 values
    X = np.loadtxt(coulomb_txtfile).reshape((-1,1,29,29)).astype(np.float32)
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



batch_size = None
epochs = None
dataDim = None 

print("Done")
meta_seed = np.random.randint(2147462579)
# meta_seed = 61178996
# meta_seed = 905679106 # kld zero
meta_seed = 2019671109 # nan
np.random.seed(meta_seed)
print("Meta_seed=%d"%meta_seed)
 
get_optimizer = {'adam':adam, 'sgd':sgd, 'momentum':momentum, 'nesterov_momentum':nesterov_momentum}

@click.command()
@click.option('--batchsize', default=100, help="Mini batch size.")
@click.option('--nepochs', default=500, help="Number of epochs for which to train the network.")
@click.option('--plotevery', default=10, help="Print test error, save params etc every --plotevery epochs.")
@click.option('--learningrate', default=0.001, help="Learning rate used by the optimizer")
@click.option('--normalizegrads', type=click.IntRange(1,10), help="normalizes gradients to have the corresponding L2-norm.")
@click.option('--clipgrads', type=click.IntRange(1,10), help="clips gradients to be +/- [value]")
@click.option('--enabledebug', is_flag=True, help="Set this flag to enable debugging dumps (mainly parameter snapshots every minibatch)")
@click.option('--optimizer', default='adam', help="The gradient update algorithm to use among {}".format(get_optimizer.keys()))
@click.argument('datadir', type=click.Path(exists=True)) # help="The folder where energies.txt and coulomb.txt can be found"
@click.argument('outputdir', type=click.Path(exists=True), default=os.getcwd()) # help="Folder where all the exeperiment artifacts are stored (Default: PWD)"
def get_options(batchsize, nepochs, plotevery, 
        learningrate, normalizegrads, 
        clipgrads, enabledebug, optimizer, datadir, outputdir):

    global batch_size;  batch_size  = batchsize
    global epochs;      epochs      = nepochs

    print("Changing pwd to {}".format(outputdir))
    os.chdir(outputdir)

    mydir = os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    os.chdir(mydir)

    app_name = sys.argv[0]
    logger = get_logger(app_name=app_name, logfolder=mydir)

    # Load dataset
    X,Y = load_data(datadir + os.sep + "coulomb.txt",
                    datadir + os.sep + "energies.txt")
    Y, Y_mean, Y_std, Y_binarized = preprocess_targets(Y)
    [X_train, X_test], [Y_train, Y_test], splits = get_data_splits(X,Y, splits=[90,10])
    [Y_binarized_train, Y_binarized_test] = np.split(Y_binarized,splits)[:-1]

    np.savez('Y_vals.npz', Y_train=Y_train, Y_test=Y_test, Y_binarized_test=Y_binarized_test, Y_binarized_train=Y_binarized_train)
    np.savez('X_vals.npz', X_train=X_train, X_test=X_test)

    dataDim = X.shape[1:]
 
    # TODO !!!!I am here
    # print("Train set size {}, Train set (labelled) size {}, Test set size {}," +
    #         "Validation set size {}".format(
    #             train_set[0].size,train_set_labeled[0].size, 
    #             test_set[0].size, valid_set[0].size))


    # Defining the model now. 
    th_coulomb      = T.ftensor4()
    th_energies     = T.fmatrix()
    th_energies_bin = T.fmatrix()

    l_input    = InputLayer(shape=(None, 1, 29,29),input_var=th_coulomb,   name="Input")
    l_conv1    = Conv2DLayer(l_input,5,3, pad="same",                     name="conv1")
    l_conv2    = Conv2DLayer(l_conv1,5,3, pad="same",                     name="conv2")
    l_maxpool1 = MaxPool2DLayer(l_conv2, (2,2),                           name="maxpool1")
    l_conv3    = Conv2DLayer(l_maxpool1, 5, 2, pad="same",                name="conv3")
    l_maxpool2 = MaxPool2DLayer(l_conv3, (2,2),                           name="maxpool2")
    l_conv4    = Conv2DLayer(l_maxpool2, 5, 2, pad="same",                name="conv4")
    l_flatten  = FlattenLayer(l_conv4,                                    name="flatten")
    l_realOut  = DenseLayer(l_flatten, num_units=20, nonlinearity=linear, name="realOut")
    l_binOut   = DenseLayer(l_flatten, num_units=20, nonlinearity=sigmoid,name="binOut")
    l_output   = ElemwiseMergeLayer([l_binOut, l_realOut], T.mul)

    energy_output = get_output(l_output)
    binary_output = get_output(l_binOut)

    loss_real   = T.sum(abs(energy_output - th_energies))
    loss_binary = T.sum(binary_crossentropy(binary_output, th_energies_bin))
    loss = loss_real + loss_binary
    
    params = get_all_params(l_output)
    grad = T.grad(loss, params)

    if normalizegrads is not None:
        grad = lasagne.updates.total_norm_constraint(grad, max_norm=normalizegrads)

    if clipgrads is not None:
        grad = [T.clip(g, -clipgrads, clipgrads) for g in grad]

    optimization_algo = get_optimizer[optimizer]
    updates = optimization_algo(grad, params, learning_rate=learningrate)
   
    train_fn  = theano.function([th_coulomb, th_energies, th_energies_bin], [loss, energy_output], updates=updates, allow_input_downcast=True)
    get_grad  = theano.function([th_coulomb, th_energies, th_energies_bin], grad)
    # get_updates = theano.function([th_data, th_labl], [updates.values()])
    val_fn    = theano.function([th_coulomb, th_energies, th_energies_bin], [loss, energy_output], updates=updates, allow_input_downcast=True)
    
    datapoints = len(X_train)
    print("datapoints = %d"%datapoints)
    
    with open(os.path.join(mydir, "data.txt"),"w") as f:
        script = app_name
        for elem in ["meta_seed", "dataDim", "batch_size", "epochs", "learningrate","normalizegrads","clipgrads","enabledebug","optimizer","script"]:
            f.write("{} : {}\n".format(elem, eval(elem)))
    
    for epoch in range(epochs):
        batch_start = 0
        train_loss = []

        indices = np.random.permutation(datapoints)
        minibatches = int(datapoints/batch_size)
        for minibatch in range(minibatches):
            train_idxs     = indices[batch_start:batch_start+batch_size]
            X_train_batch  = X_train[train_idxs,:]
            Yr_train_batch = Y_train[train_idxs,:]
            Yb_train_batch = Y_binarized_train[train_idxs, :]

            train_output = train_fn(X_train_batch, Yr_train_batch, Yb_train_batch)
            batch_start  = batch_start + batch_size
            
            train_loss.append(train_output[0])

            if enabledebug:
                # Debugging information
                batchIdx = epoch*minibatches+minibatch
                fn = 'params_{:>010d}'.format() # saving params
                param_values = get_all_param_values(l_output)
                param_norm   = np.linalg.norm(np.hstack([param.flatten() for param in param_values]))
                gradients = get_grad(X_train_batch, Yr_train_batch, Yb_train_batch)
                gradient_norm = np.linalg.norm(np.hstack([gradient.flatten() for gradient in gradients]))
                logger.debug("Epoch : {:0>4}  minibatch {:0>3} Gradient Norm : {:>0.4}, Param Norm : {:>0.4} GradNorm/ParamNorm : {:>0.4} (Values from Prev. Minibatch) Train loss {}".format(epoch, minibatch, gradient_norm, param_norm, gradient_norm/param_norm,train_loss[-1]))
                param_names  = [param.__str__() for param in get_all_params(l_output)]
                np.savez(fn + '.npz', **dict(zip(param_names, param_values)))
                np.savez('Y_train_pred_{}.npz'.format(batchIdx), Y_train_pred = train_output[1])
                if np.isnan(gradient_norm):
                    pdb.set_trace()
                


        if(epoch % plotevery == 0):
            logger.info("Epoch {} of {}".format(epoch, epochs))

            fn = 'params_{:>03d}'.format(epoch) # saving params
            param_values = get_all_param_values(l_output)
            param_norm   = np.linalg.norm(np.hstack([param.flatten() for param in param_values]))
            param_names  = [param.__str__() for param in get_all_params(l_output)]
            if not enabledebug:
                np.savez(fn + '.npz', **dict(zip(param_names, param_values)))
                np.savez('Y_train_pred_{}.npz'.format(epoch), Y_train_pred = train_output[1])


            gradients = get_grad(X_train_batch, Yr_train_batch, Yb_train_batch)
            gradient_norm = np.linalg.norm(np.hstack([gradient.flatten() for gradient in gradients]))
            logger.info("  Gradient Norm : {}, Param Norm : {} GradNorm/ParamNorm : {} ".format(gradient_norm, param_norm, gradient_norm/param_norm))
            logger.info("  Train loss {:>0.4}".format(np.mean(train_loss)))
            
            test_loss, test_prediction = val_fn(X_test, Y_test, Y_binarized_test)
            np.savez('Y_test_pred_{}.npz'.format(epoch), Y_test_pred = test_prediction)
            logger.info("  Test loss {}".format(test_loss))
            
if __name__ == "__main__":
    get_options()

