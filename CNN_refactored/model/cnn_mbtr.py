import os
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import logging
import numpy as np
import lasagne
import theano.tensor as T
import theano
# from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
# from lasagne.layers.dnn import MaxPool2DDNNLayer as Conv2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DenseLayer, FlattenLayer

logger = logging.getLogger(__name__)

def cnn_mbtr(filters_list, outdim, cost, input_dims = None, activation="rectify", **kwargs):

    # path to targets_file is not NONE
    # sym_coulomb = T.imatrix()
    sym_coulomb = T.fmatrix()
    sym_y = T.fmatrix()
    sym_learn_rate = T.scalar()

    try:
        nonlinearity = getattr(lasagne.nonlinearities, activation)
    except AttributeError as e:
        print(e)
        raise RuntimeError("Activation {} missing in lasagne.nonlinearities.".format(activation))
    
    # layer_input_dims = (None, *input_dims) # (None, 1, 23, 23) if input_dims == (1, 23, 23)
    layer_input_dims = [None]
    layer_input_dims.extend(input_dims)
    
    layers = []
    layers.append(lasagne.layers.InputLayer(layer_input_dims, name="layer_input"))

    for idx, num_filters in enumerate(filters_list):
            layers.append(Conv2DLayer(layers[-1],
                num_filters = num_filters,
                filter_size = (5,1),
                pad = "same",
                flip_filters = False,
                nonlinearity = nonlinearity,
                name="layer_conv_{}_1".format(idx)
                ))
            layers.append(Conv2DLayer(layers[-1],
                num_filters = num_filters,
                filter_size = (3,1),
                pad = "same",
                flip_filters = False,
                nonlinearity = nonlinearity,
                name="layer_conv_{}_2".format(idx)
                ))
            layers.append(Conv2DLayer(layers[-1],
                num_filters = num_filters,
                filter_size = (3,1),
                pad = "same",
                flip_filters = False,
                nonlinearity = nonlinearity,
                name="layer_conv_{}_3".format(idx)
                ))
            layers.append(MaxPool2DLayer(layers[-1],
                pool_size = 2,
                name="layer_maxpool_1"
                ))
    
    layers.append(FlattenLayer(layers[-1]))
    layers.append(DenseLayer(layers[-1], num_units = outdim, nonlinearity=lasagne.nonlinearities.linear))
    l_out = layers[-1] 

    # l_in_Z = lasagne.layers.InputLayer((None, max_mol_size))
    # l_in_D = lasagne.layers.InputLayer((None, max_mol_size, max_mol_size, num_dist_basis))
    # l_mask = MaskLayer(l_in_Z)
    # l_c0 = SwitchLayer(l_in_Z, num_species, c_len, W=lasagne.init.Uniform(1.0/np.sqrt(c_len)))

    # l_cT = RecurrentLayer(l_c0, l_in_D, l_mask, num_passes=num_interaction_passes, num_hidden=num_hidden_neurons)

    # # Compute energy contribution from each atom
    # l_atom1 = lasagne.layers.DenseLayer(l_cT, 15, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    # l_atom2 = lasagne.layers.DenseLayer(l_atom1, values_to_predict, nonlinearity=None, num_leading_axes=2) # outdim (-1, 23, values_to_predict)
    # l_atomE = lasagne.layers.ExpressionLayer(l_atom2, lambda x: (x*Estd+Emean)) # Scale and shift by mean and std deviation
    # l_mask = lasagne.layers.ReshapeLayer(l_mask, ([0], [1], 1)) # add an extra dimension so that l_atomE (-1, 23, 16) l_mask "after reshape" (-1, 23, 1) can be multiplied
    # l_out = SumMaskedLayer(l_atomE, l_mask)

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        logger.debug("%s, %s" % (p, p.get_value().shape))

    # out_train = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=False)
    # out_test = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=True)
    out_train = lasagne.layers.get_output(l_out, {layers[0] : sym_coulomb}, deterministic=False)
    out_test = lasagne.layers.get_output(l_out, {layers[0] : sym_coulomb}, deterministic=True)
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
            inputs = [sym_coulomb, sym_y, sym_learn_rate],
            outputs = cost_train,
            updates = updates
            )

    f_eval_test = theano.function(
            inputs = [sym_coulomb],
            outputs = out_test
            )

    f_test = theano.function(
            inputs = [sym_coulomb, sym_y],
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
