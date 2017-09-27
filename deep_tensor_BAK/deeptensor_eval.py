import sys
import pdb
from pprint import pprint
import gzip
import cPickle as pickle
import lasagne
import numpy as np
import theano.tensor as T
import theano
from utils import feature_expand
from dtnn_layers import SwitchLayer, MaskLayer, SumMaskedLayer, RecurrentLayer

theano.config.floatX = 'float32'

def make_scalar_if_len1(np_loadz_dict):
    """
    Input: dictionary obtained from numpy.load
    Output: dictionary with values having length 1 converted to scalars
    
    when saving scalars using numpy savez it converts these to array(scalar)
    consequently when using these values (after loading from the npz file)
    theano dimensions having this array(scalar) causes dimension mismatch error.
    """
    pass


def build_model(max_mol_size,Estd,Emean,num_dist_basis,num_species,c_len, num_hidden_neurons,
        num_interaction_passes,values_to_predict,mae_cost=False,**kwargs):

    max_mol_size = np.int32(max_mol_size)
    num_dist_basis = np.int32(num_dist_basis)
    num_hidden_neurons = np.int32(num_hidden_neurons)
    values_to_predict = np.int32(values_to_predict)
    num_species = np.int32(num_species)

    print("max_mol_size ={}".format(max_mol_size))
    print("num_dist_basis ={}".format(num_dist_basis))
    print("num_hidden_neurons ={}".format(num_hidden_neurons))
    print("values_to_predict ={}".format(values_to_predict))
    print("num_species ={}".format(num_species))


    sym_Z = T.imatrix()
    sym_D = T.tensor4()
    # sym_y = T.vector()
    # if path_to_targets_file > 1:
    #     sym_y = T.fmatrix()
    ## We always predict either 16,20 or 300 values so.
    sym_y = T.fmatrix()
    sym_learn_rate = T.scalar()

    l_in_Z = lasagne.layers.InputLayer((None, max_mol_size))
    l_in_D = lasagne.layers.InputLayer((None, max_mol_size, max_mol_size, num_dist_basis))
    l_mask = MaskLayer(l_in_Z)
    l_c0 = SwitchLayer(l_in_Z, num_species, c_len, W=lasagne.init.Uniform(1.0/np.sqrt(c_len)))

    l_cT = RecurrentLayer(l_c0, l_in_D, l_mask, num_passes=num_interaction_passes, num_hidden=num_hidden_neurons)

    # Compute energy contribution from each atom
    l_atom1 = lasagne.layers.DenseLayer(l_cT, 15, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom2 = lasagne.layers.DenseLayer(l_atom1, values_to_predict, nonlinearity=None, num_leading_axes=2) # outdim (-1, 23, values_to_predict)
    #if path_to_targets_file is None:
    #    l_atom2 = lasagne.layers.FlattenLayer(l_atom2, outdim=2) # Flatten singleton dimension # outdim (-1, 23)
        # but if path_to_targets is not None, then we don't want to flatten since we want to get outputs (energies, or spectrum values) for each atom: ie. we want outdim (-1, 23, values_to_predict)
    l_atomE = lasagne.layers.ExpressionLayer(l_atom2, lambda x: (x*Estd+Emean)) # Scale and shift by mean and std deviation
    #if path_to_targets_file is not None:
    #    l_mask = lasagne.layers.ReshapeLayer(l_mask, ([0], [1], 1)) # add an extra dimension so that l_atomE (-1, 23, 16) l_mask "after reshape" (-1, 23, 1) can be multiplied
    l_mask = lasagne.layers.ReshapeLayer(l_mask, ([0], [1], 1)) # add an extra dimension so that l_atomE (-1, 23, 16) l_mask "after reshape" (-1, 23, 1) can be multiplied
    l_out = SumMaskedLayer(l_atomE, l_mask) # TODO : BUG HERE.

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        print("%s, %s" % (p, p.get_value().shape))

    out_train = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=False)
    out_test = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=True)
    if mae_cost is True:
        cost_train = T.mean(np.abs(out_train-sym_y))
        cost_test = T.mean(np.abs(out_test-sym_y))
        print("Used MAE cost")
    else:
        cost_train = T.mean(lasagne.objectives.squared_error(out_train, sym_y))
        cost_test = T.mean(lasagne.objectives.squared_error(out_test, sym_y))
        print("Used MSE cost")


    updates = lasagne.updates.adam(cost_train, params, learning_rate=sym_learn_rate)


    f_train = theano.function(
            inputs = [sym_Z, sym_D, sym_y, sym_learn_rate],
            outputs = cost_train,
            updates = updates
            )

    f_eval_test = theano.function(
            inputs = [sym_Z, sym_D],
            outputs = out_test
            )

    f_test = theano.function(
            inputs = [sym_Z, sym_D, sym_y],
            outputs = cost_test,
            )

    return f_train, f_eval_test, f_test, l_out

def load_params_from_epoch(epoch, l_out,artifacts_dir="."):
    model_params = None
    with gzip.open(artifacts_dir+"/results/model_epoch%d.pkl.gz" % epoch, "rb") as f:
        model_params = pickle.load(f)
    load_model_params(model_params, l_out)

def load_model_params(model_params , l_out):
    lasagne.layers.set_all_param_values(l_out, model_params)

#def main(Z,D,Y,Emean,Estd,hyperparams,artifacts_dir=".", val_or_test="val",epoch=0,cost_type="mse"):
def main(Z,D,Y,f_evaluate,l_out,artifacts_dir=".",epoch=0):
    load_params_from_epoch(artifacts_dir=artifacts_dir, epoch=epoch, l_out=l_out) 
    # print("Z shape = {}".format(Z.shape))
    # print("D shape = {}".format(D.shape))
    #return f_eval_test(Z,D) - Y
    return f_evaluate(Z,D)-Y

def init(artifacts_dir=".",cost_type="mse"):
    X_data = np.load(artifacts_dir+"/X_vals.npz")
    Y_data = np.load(artifacts_dir+"/Y_vals.npz")

    if val_or_test == "val":
        Z = X_data["Z_val"]
        D = X_data["D_val"]
        Y = Y_data["Y_val"]
    else:
        Z = X_data["Z_test"]
        D = X_data["D_test"]
        Y = Y_data["Y_test"]
    
    Emean = Y_data["Y_mean"]
    Estd  = Y_data["Y_std"]
    del Y_data
    del X_data
    
    # load hyper params
    hyperparams = np.load(artifacts_dir+"/results/hyperparams.npz")    
    #pprint(hyperparams.items())

    num_dist_basis,mu_max = hyperparams["num_dist_basis"], hyperparams["mu_max"]
    print("num_dist_basis {}, mu_max = {}".format(num_dist_basis, mu_max))
    D = feature_expand(D,num_dist_basis, mu_max=mu_max)

    mae_cost = True if cost_type == "mae" else False
    f_train, f_eval_test, f_test, l_out = build_model(mae_cost=mae_cost,**hyperparams)

    #return Z,D,Y,Emean,Estd,hyperparams,f_train,f_eval_test,f_test,l_out
    return f_train,f_eval_test,f_test,l_out,Z,D,Y

def aggregate_cost(mse_or_mae,cost):
    if mse_or_mae == "mse":
        cost = np.mean(np.square((cost)))
    else:
        cost = np.mean(np.abs(cost))
    return cost

if __name__ == "__main__":
    artifacts_dir=sys.argv[1]
    val_or_test = sys.argv[2]
    max_epochs = 10000
    mse_or_mae = sys.argv[3]
    
    f_train,f_eval_test,f_test,l_out,Z,D,Y = init(artifacts_dir=artifacts_dir,cost_type=mse_or_mae)
    f_evaluate = f_eval_test
    if "epoch" in sys.argv[4:]:
        epoch_idx = sys.argv.index("epoch")+1
        epoch = int(sys.argv[epoch_idx])
        cost = main(Z,D,Y,f_evaluate, l_out, artifacts_dir=artifacts_dir,epoch=epoch)
        print("cost = {} epoch = {}".format(aggregate_cost(mse_or_mae, cost), epoch))
        filename=artifacts_dir+"/Y_pred_{}_epoch_{}.npz".format(val_or_test,epoch)
        np.savez(filename, y_pred=f_evaluate(Z,D), y_target=Y)
        print("y_pred,y_{} saved as {}".format(val_or_test,filename))


    else:
        lowest_cost = np.inf
        for epoch in range(0,max_epochs,2):
            try:
                #cost = main(Z,D,Y,f_eval_test, l_out, artifacts_dir=artifacts_dir,epoch=epoch)
                cost = main(Z,D,Y,f_eval_test, l_out, artifacts_dir=artifacts_dir,epoch=epoch)
            except IOError as e:
                print("Couldn't load model file. epoch {}, e = {}".format(epoch,e))
            else:
                cost = aggregate_cost(mse_or_mae,cost) 
                if cost < lowest_cost:
                    lowest_cost = cost
                    print("Lowest cost {} found in epoch {}".format(cost,epoch))

                print("Epoch {} error = {}".format(epoch, cost))
