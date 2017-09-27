import os
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import pdb
import logging
import numpy as np
import lasagne
import theano.tensor as T
import theano
# from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
# from lasagne.layers.dnn import MaxPool2DDNNLayer as Conv2DLayer
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DenseLayer #, FlattenLayer

logger = logging.getLogger(__name__)

def orig_model(units_list, outdim, cost, input_dim, activation="sigmoid", **kwargs):
    #Emean, Estd, max_mol_size, num_dist_basis, c_len, num_species,
    #    num_interaction_passes, num_hidden_neurons, values_to_predict,cost):

    # path to targets_file is not NONE
    # sym_coulomb = T.imatrix()
    sym_X = T.fmatrix()
    sym_y = T.fmatrix()
    sym_learn_rate = T.scalar()

    logger.debug("Using activation : {}".format(activation))

    try:
        nonlinearity = getattr(lasagne.nonlinearities, activation)
    except AttributeError as e:
        print(e)
        raise RuntimeError("Activation {} missing in lasagne.nonlinearities.".format(activation))
    
    # layer_input_dims = (None, *input_dims) # (None, 1, 23, 23) if input_dims == (1, 23, 23)
    layer_input_dims = [None]
    layer_input_dims.append(input_dim)
    
    layers = []
    layers.append(lasagne.layers.InputLayer(layer_input_dims, name="layer_input"))

    for idx, num_units in enumerate(units_list):
            layers.append(DenseLayer(layers[-1],
                num_units = num_units,
                nonlinearity = nonlinearity,
                name="layer_{}".format(idx)
                ))
    
    # layers.append(FlattenLayer(layers[-1]))
    layers.append(DenseLayer(layers[-1], num_units = outdim, nonlinearity=nonlinearity))
    l_out = layers[-1] 

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        logger.debug("%s, %s" % (p, p.get_value().shape))

    # out_train = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=False)
    # out_test = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=True)
    out_train = lasagne.layers.get_output(l_out, {layers[0] : sym_X}, deterministic=False)
    out_test = lasagne.layers.get_output(l_out, {layers[0] : sym_X}, deterministic=True)
    if cost == "mae":
        cost_train = T.mean(np.abs(out_train-sym_y))
        cost_test = T.mean(np.abs(out_test-sym_y))
        logger.info("Used MAE cost")
    elif cost == "rmse":
        cost_train = T.mean(lasagne.objectives.squared_error(out_train, sym_y))
        cost_test = T.mean(lasagne.objectives.squared_error(out_test, sym_y))
        logger.info("Used MSE cost")
    else:
        raise ValueError("unknown cost function {}".format(cost))

    updates = lasagne.updates.adam(cost_train, params, learning_rate=sym_learn_rate)

    f_train = theano.function(
            inputs = [sym_X, sym_y, sym_learn_rate],
            outputs = cost_train,
            updates = updates
            )

    f_eval_test = theano.function(
            inputs = [sym_X],
            outputs = out_test
            )

    f_test = theano.function(
            inputs = [sym_X, sym_y],
            outputs = cost_test,
            )



    # f_train = theano.function(
    #         inputs = [sym_Z, sym_D, sym_y, sym_learn_rate],
    #         outputs = cost_train,
    #         updates = updates
    #         )

    # f_eval_test = theano.function(
    #         inputs = [sym_Z, sym_D],
    #         outputs = out_test
    #         )

    # f_test = theano.function(
    #         inputs = [sym_Z, sym_D, sym_y],
    #         outputs = cost_test,
    #         )

    return f_train, f_eval_test, f_test, l_out


if __name__ == "__main__":
    # write tests here
    pass
