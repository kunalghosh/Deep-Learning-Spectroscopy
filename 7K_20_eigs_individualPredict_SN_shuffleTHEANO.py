# coding: utf-8

# Takes in the coulomb matrix and predicts the 20 eigen values
# individually

# In[3]:

from __future__ import print_function

import os
import sys
from datetime import datetime
import numpy as np
import pdb
import logging

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
from coulomb_shuffle import coulomb_shuffle

logger = None

def load_data(coulomb_txtfile, energies_txtfile):
    Y = np.loadtxt(energies_txtfile).astype(np.float32) # Note that here y_i is a vector of 20 values
    X = np.loadtxt(coulomb_txtfile).reshape((-1,29,29)).astype(np.float32)
    return X,Y

def preprocess_targets(Y, binary_threshold=10**-5, zero_mean=True, unit_var=True):
    """
    zero mean and unit variance
    """
    Y_binarized = (Y >= binary_threshold).astype(np.float32)
    _mean = Y.mean(axis=0)
    if zero_mean == False:
        """
        Don't make Y zero mean. But _mean still returned so that other 
        code. Mainly visualization code. Doesn't break.
        """
        logger.info("[Preprocessing] Y not zero mean")
        _mean = np.zeros_like(_mean)
    _std  = Y.std(axis=0)
    if unit_var == False:
        """
        Same reason as in zero mean
        """
        logger.info("[Preprocessing] Y not unit variance")
        _std = np.ones_like(_std)
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

def make_shared(data, borrow=True):
    shared_data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=borrow)
    return shared_data

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
@click.option('--learningrate', help="Learning rate used by the optimizer")
@click.option('--normalizegrads', type=click.IntRange(1,10), help="normalizes gradients to have the corresponding L2-norm.")
@click.option('--clipgrads', type=click.IntRange(1,10), help="clips gradients to be +/- [value]")
@click.option('--enabledebug', is_flag=True, help="Set this flag to enable debugging dumps (mainly parameter snapshots every minibatch)")
@click.option('--optimizer', default='adam', help="The gradient update algorithm to use among {}".format(get_optimizer.keys()))
@click.option('--yzeromean', is_flag=True, help="Set this flag to make Y zero mean.")
@click.option('--yunitvar', is_flag=True, help="Set this flag to make Y unit variance.")
@click.argument('datadir', type=click.Path(exists=True)) # help="The folder where energies.txt and coulomb.txt can be found"
@click.argument('outputdir', type=click.Path(exists=True), default=os.getcwd()) # help="Folder where all the exeperiment artifacts are stored (Default: PWD)"
def get_options(batchsize, nepochs, plotevery, 
        learningrate, normalizegrads, 
        clipgrads, enabledebug, optimizer, yzeromean, yunitvar, datadir, outputdir):

    global batch_size;  batch_size  = batchsize
    global epochs;      epochs      = nepochs

    print("Changing pwd to {}".format(outputdir))
    os.chdir(outputdir)

    mydir = os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    os.chdir(mydir)

    app_name = sys.argv[0]
    global logger
    logger = get_logger(app_name=app_name, logfolder=mydir)

    # Load dataset
    X,Y = load_data(datadir + os.sep + "coulomb.txt",
                    datadir + os.sep + "energies.txt")
    Y, Y_mean, Y_std, Y_binarized = preprocess_targets(Y, zero_mean=yzeromean, unit_var=yunitvar)
    [X_train, X_test], [Y_train, Y_test], splits = get_data_splits(X,Y, splits=[90,10])
    [Y_binarized_train, Y_binarized_test] = np.split(Y_binarized,splits)[:-1]

    np.savez('Y_vals.npz', Y_train=Y_train, Y_test=Y_test, Y_binarized_test=Y_binarized_test, Y_binarized_train=Y_binarized_train, Y_mean=Y_mean, Y_std=Y_std)
    np.savez('X_vals.npz', X_train=X_train, X_test=X_test)

    dataDim = X.shape[1:]
    outputDim = Y.shape[1]
    datapoints = len(X_train)
    print("datapoints = %d" % datapoints)

    # # making the datapoints shared variables
    # X_train           = make_shared(X_train)
    # X_test            = make_shared(X_test)
    # Y_train           = make_shared(Y_train)
    # Y_test            = make_shared(Y_test)
    # Y_binarized_train = make_shared(Y_binarized_train)
    # Y_binarized_test  = make_shared(Y_binarized_test)
 
    # TODO !!!!I am here
    # print("Train set size {}, Train set (labelled) size {}, Test set size {}," +
    #         "Validation set size {}".format(
    #             train_set[0].size,train_set_labeled[0].size, 
    #             test_set[0].size, valid_set[0].size))

    eigen_value_count = 20

    # Defining the model now. 
    th_coulomb      = T.ftensor3()
    th_energies     = T.fmatrix()
    th_energies_bin = T.fmatrix()
    th_learningrate = T.fscalar()

    l_input    = InputLayer(shape=(None, 29,29),input_var=th_coulomb,     name="Input")
    l_input    = FlattenLayer(l_input,                                       name="FlattenInput")
    
    l_pseudo_bin = []; l_h1 = []; l_h2 = []; l_realOut = []; l_binOut = [];

    for branch_num in range(eigen_value_count):
        l_pseudo_bin.append(DenseLayer(l_input, num_units=2000, nonlinearity=sigmoid, name="PseudoBinarized_%d" % branch_num))
        l_h1.append(DenseLayer(l_pseudo_bin, num_units=1000, nonlinearity=rectify, name="hidden_1_%d" % branch_num))
        l_h2.append(DenseLayer(l_h1[-1], num_units=400,  nonlinearity=rectify, name="hidden_2_%d" % branch_num))
        l_realOut.append(DenseLayer(l_h2[-1],    num_units=1,    nonlinearity=linear,  name= "realOut_%d" % branch_num))
        l_binOut.append(DenseLayer(l_h2[-1],num_units=1, nonlinearity=sigmoid,name="binOut"))
        
    l_realOut_cat = ConcatLayer(l_realOut, name="real_concat") 
    l_binOut_cat  = ConcatLayer(l_binOut,  name="bin_concat") 
    l_output = ElemwiseMergeLayer([l_binOut_cat, l_realOut_cat], T.mul, name="final_output")

    energy_output = get_output(l_output)
    binary_output = get_output(l_binOut_cat)

    loss_real   = T.mean(abs(energy_output - th_energies))
    loss_binary = T.mean(binary_crossentropy(binary_output, th_energies_bin))
    loss = loss_real + loss_binary
    
    params = get_all_params(l_output)
    grad = T.grad(loss, params)

    if normalizegrads is not None:
        grad = lasagne.updates.total_norm_constraint(grad, max_norm=normalizegrads)

    if clipgrads is not None:
        grad = [T.clip(g, -clipgrads, clipgrads) for g in grad]

    optimization_algo = get_optimizer[optimizer]
    # updates = optimization_algo(grad, params, learning_rate=learningrate)
    updates = optimization_algo(grad, params, learning_rate=th_learningrate)
   
    train_fn  = theano.function([th_coulomb, th_energies, th_energies_bin, th_learningrate], [loss, energy_output], updates=updates, allow_input_downcast=True)
    get_grad  = theano.function([th_coulomb, th_energies, th_energies_bin], grad)
    # get_updates = theano.function([th_data, th_labl], [updates.values()])
    # val_fn    = theano.function([th_coulomb, th_energies, th_energies_bin], [loss, energy_output], updates=updates, allow_input_downcast=True)
    val_fn    = theano.function([th_coulomb, th_energies, th_energies_bin], [loss, energy_output], allow_input_downcast=True)
    
    datapoints = len(X_train)
    print("datapoints = %d"%datapoints)
    
    with open(os.path.join(mydir, "data.txt"),"w") as f:
        script = app_name
        for elem in ["meta_seed", "dataDim", "batch_size", "epochs", "learningrate","normalizegrads","clipgrads","enabledebug","optimizer","plotevery","script"]:
            f.write("{} : {}\n".format(elem, eval(elem)))
    
    train_loss_lowest = np.inf
    test_loss_lowest = np.inf

    row_norms = np.linalg.norm(X_train, axis=-1)
    for epoch in range(epochs):
        batch_start = 0
        train_loss = []

        if learningrate == None:
            if epoch < 50:
                learning_rate = 0.0001
            elif epoch < 100:
                learning_rate = 0.00001
            elif epoch < 500:
                learning_rate = 0.000001
            else:
                learning_rate = 0.0000001
        else:
            learning_rate = eval(learningrate)
            if isinstance(learning_rate, float):
                pass
            elif isinstance(learning_rate, list):
                for epch, lrate in learning_rate:
                    # ensure that last epoch is float("inf")
                    if epoch <= epch:
                        learning_rate = lrate
                        break
            else:
                raise RuntimeError("Invalid learning rate.Either \n 1) Float or 2) List [[epch, lrate],...,[float('inf'), lrate]]")



        indices = np.random.permutation(datapoints)
        minibatches = int(datapoints/batch_size)
        logger.debug("Shuffling Started.")
        X_train = coulomb_shuffle(X_train, row_norms)
        logger.debug("Shuffling complete.")
        for minibatch in range(minibatches):
            train_idxs     = indices[batch_start:batch_start+batch_size]
            X_train_batch  = X_train[train_idxs,:]
            Yr_train_batch = Y_train[train_idxs,:]
            Yb_train_batch = Y_binarized_train[train_idxs, :]

            train_output = train_fn(X_train_batch, Yr_train_batch, Yb_train_batch, learning_rate)
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
                if train_loss[-1] < train_loss_lowest:
                    train_loss_lowest = train_loss[-1]
                    np.savez('Y_train_pred_best.npz', Y_train_pred = train_output[1])
                    logger.debug("Found the best training prediction (Y_train_pred_best) at %d epoch %d minibatch" % (epoch, minibatch))
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
                mean_train_loss = np.mean(train_loss)
                if mean_train_loss < train_loss_lowest:
                    train_loss_lowest = mean_train_loss
                    np.savez('Y_train_pred_best.npz', Y_train_pred = train_output[1])
                    logger.info("Found the best training prediction (Y_train_pred_best) at %d epoch" % epoch)


            gradients = get_grad(X_train_batch, Yr_train_batch, Yb_train_batch)
            gradient_norm = np.linalg.norm(np.hstack([gradient.flatten() for gradient in gradients]))
            logger.info("  Gradient Norm : {:>0.4}, Param Norm : {:>0.4} GradNorm/ParamNorm : {:>0.4} ".format(gradient_norm, param_norm, gradient_norm/param_norm))
            logger.info("  Train loss {:>0.4}".format(np.mean(train_loss)))
            
            test_loss, test_prediction = val_fn(X_test, Y_test, Y_binarized_test)
            np.savez('Y_test_pred_{}.npz'.format(epoch), Y_test_pred = test_prediction)
            logger.info("  Test loss {}".format(test_loss))
            if test_loss < test_loss_lowest:
               test_loss_lowest = test_loss 
               np.savez('Y_test_pred_best.npz', Y_test_pred = test_prediction)           
               logger.info("Found the best test prediction (Y_test_pred_best) at %d epoch" % epoch)
if __name__ == "__main__":
    get_options()

