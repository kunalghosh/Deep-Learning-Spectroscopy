import gc
import os
import sys
import copy
from os import path
import pdb
import cPickle as pickle
import logging
import gzip
import timeit
import tables
import numpy as np
import lasagne
import theano.tensor as T
import theano
# from utils import load_qm7b_data, load_oqmd_data, feature_expand
from sklearn.model_selection import train_test_split
from datetime import datetime
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from util import get_logger
# from model.deeptensor import orig_model
# import model.deeptensor as model_factory
import model as model_factory
import click
from coulomb_shuffle import coulomb_shuffle

theano.config.floatX = 'float32'

def train_and_get_error(train_data, valid_data, test_data, Estd, Emean, 
        conv_filters, values_to_predict, max_epochs, batch_size, cost, 
        model_name, learn_rate, earlystop_epochs, logger, data_dim, 
        check_every=2,**kwargs):
        # training_data,valid_data,test_data, trainin_hyperparams,model_name, model_hyperparams,
        #                 num_train_batches,num_train_samples,batch_size = 100,max_epochs=10000,
        #                 c_len = 30,num_hidden_neurons = 60,num_interaction_passes = 2,values_to_predict=-1
        #                 ):

    """
    param check_every      : Check the validation and test error every, these many epochs.
    param earlystop_epochs : If the validation error has not reduced in these many epochs
                             the training stops. It is decremented by 'check_every' depending
                             on the 'cost'. It is reset to its initial value if error reduces.
    """
    X_train, y_train = train_data
    X_val, y_val = valid_data
    X_test, y_test    = test_data

    rng = np.random.RandomState(4)
    np.random.seed(1)

    try:
        logger.info("Trying to load model '%s'" % model_name)
        model = getattr(model_factory, model_name)
    except AttributeError as e:
        logger.error("Model not found. %s" % e)
        RuntimeError("Couldn't find model %s in the model_factory" % model_name)
    else:
        logger.info("Loaded '%s'" % model_name)
    
    f_train, f_eval_test, f_test, l_out =  model(filters_list = conv_filters,
            outdim = values_to_predict, cost = cost, input_dims = data_dim, **kwargs)
        
    start_time = timeit.default_timer()

    lowest_test_mae = np.inf
    lowest_test_rmse = np.inf
    lowest_test_error = np.inf
    # mu_max = None#np.max(D_train)+1

    # D_val_fe = feature_expand(D_val, num_dist_basis, mu_max=mu_max)
    # D_test_fe = feature_expand(D_test, num_dist_basis,mu_max=mu_max)

    # logger.info("Expanded Test and Val.")

    num_train_samples = X_train.shape[0]
    num_train_batches = num_train_samples // batch_size

    earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)


    # extend_dims = [-1]
    # extend_dims.extend(coulomb_dims)
    X_train = X_train.reshape(-1, 1, data_dim)
    X_val = X_val.reshape(-1, 1, data_dim)
    X_test = X_test.reshape(-1, 1, data_dim)
    logger.info("New X_train shape = {}".format(X_train.shape))
    # # pdb.set_trace()
    # X_train_row_norms = np.linalg.norm(X_train, axis=1)

    for epoch in range(max_epochs):
        # # Randomly shuffle coulomb matrix
        # X_train = X_train.reshape(-1, coulomb_dims[1], coulomb_dims[2])
        # X_train = coulomb_shuffle(X_train, X_train_row_norms)
        # X_train = X_train.reshape(extend_dims)

        # Randomly permute training data
        rand_perm = rng.permutation(X_train.shape[0])
        X_train_perm = X_train[rand_perm]
        # D_train_perm = D_train[rand_perm]
        y_train_perm = y_train[rand_perm]

        # logger.info("Shuffled train data")
        if learn_rate is None:
            if epoch < 10:
            #if epoch < 50:
                learning_rate = 0.01
            elif epoch < 500:
                learning_rate = 0.001
            elif epoch < 5000:
                learning_rate = 0.0001
            else:
                learning_rate = 0.00001
        else:
            learning_rate = learn_rate

        train_cost = 0
        for batch in range(num_train_batches):
            train_cost += f_train(
                    X_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    # feature_expand(D_train_perm[batch*batch_size:((batch+1)*batch_size)], num_dist_basis, mu_max=mu_max),
                    y_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    learning_rate
                    )
            #print("miniBatch %d of %d done." % (batch, num_train_batches))
        train_cost = train_cost / num_train_batches
        # logger.info("Got training cost.")

        if (epoch % check_every) == 0:
            # y_pred = f_eval_test(Z_val, D_val_fe)
            y_pred = f_eval_test(X_val)
            val_errors = y_pred-y_val

            # logger.info("Got Y val_pred")

            y_pred = f_eval_test(X_test)
            test_errors = y_pred-y_test
            test_cost = f_test(X_test, y_test)
            
            rmse = lambda x: np.sqrt(np.square(x).mean())
            mae  = lambda x: np.abs(x).mean()

            logger.info("VAL MAE:  %5.2f kcal/mol TEST MAE:  %5.2f kcal/mol" %
                    (mae(val_errors), mae(test_errors)))
            logger.info("VAL RMSE: %5.2f kcal/mol TEST RMSE: %5.2f kcal/mol" %
                    (rmse(val_errors), rmse(test_errors)))

            all_params = lasagne.layers.get_all_param_values(l_out)
            with gzip.open('results/model_epoch%d.pkl.gz' % (epoch), 'wb') as f:
                pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)

            new_test_mae = mae(test_errors)
            if  new_test_mae < lowest_test_mae:
                lowest_test_mae = new_test_mae
                logger.info("Found best test MAE : {}".format(lowest_test_mae))
                np.savez("Y_test_pred_best_mae.npz", Y_test_pred = y_pred)
                if cost == "mae":
                    # Only update early stoppping counter if we are interested in this cost
                    earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)
                    lowest_test_error = lowest_test_mae

            new_test_rmse = rmse(test_errors)
            if new_test_rmse < lowest_test_rmse:
                lowest_test_rmse = new_test_rmse
                logger.info("Found best test RMSE : {}".format(lowest_test_rmse))
                np.savez("Y_test_pred_best_rmse.npz", Y_test_pred = y_pred)
                if cost == "rmse":
                    # Only update early stoppping counter if we are interested in this cost
                    earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)
                    lowest_test_error = lowest_test_rmse

            end_time = timeit.default_timer()

            test_error = np.sqrt(test_cost)
            train_error = np.sqrt(train_cost)
            logger.debug("Time %4.1f, Epoch %4d, train_cost=%5g, test_error=%5g" % (end_time - start_time, epoch, train_error, test_error))
            start_time = timeit.default_timer()

            # update and check the early stop counter.
            earlystop_epoch_counter -= check_every
            if earlystop_epoch_counter <= 0:
                break

    # return test_error
    return lowest_test_error

def get_data(path_to_coulomb_file, path_to_targets_file, logger,  
            data_dim, train_split=0.9, valid_test_split=0.5, 
            dtype=theano.config.floatX, **kwargs):

    # Z, D, y, num_species = load_qm7b_data(num_dist_basis, dtype=theano.config.floatX,
    #                                          xyz_file=path_to_xyz_file,expand_features=expand_features)

    # X,y = load_data(coulomb_txtfile=path_to_coulomb_file,coulomb_dims = (1, coulomb_dim, coulomb_dim), dtype=theano.config.floatX)

    ## data_file = np.load(path_to_coulomb_file)
    ## data_file_name = data_file.files[0]
    ## extend_dims = [-1]
    ## extend_dims.extend(coulomb_dims)
    ## X = data_file[data_file_name].reshape(extend_dims).astype(dtype)

    # --- new loader

    try:
        # NOTE: loading as npz first and then as txt 
        # is opposite to what was done for loading 'y' values
        # loading npz first was done much later, I thought this was better.
        data_file = np.load(path_to_coulomb_file)
        data_file_name = data_file.files[0]
        X = data_file[data_file_name]
    except IOError as e:
        # can't find the npz file or
        # can't load the file as pickle file
        if path_to_coulomb_file.endswith("h5"):
            h5file = tables.open_file(path_to_coulomb_file, mode="r")
            X = h5file.root.data
        else:
            X = np.loadtxt(path_to_coulomb_file)

    # extend_dims = [-1]
    # extend_dims.extend(coulomb_dims)
    # # X = data_file[data_file_name].reshape(extend_dims).astype(dtype)
    # X = X.reshape(extend_dims).astype(dtype)
    if isinstance(X, np.ndarray): 
        X = X.astype(dtype)

    # X = data_file[data_file_name].astype(dtype)
    logger.info("X shape = {}".format(X.shape))
    # The y values are free_energies but we want to predict 16 energies / spectrum so 
    # we load them next.

    # # Largest interatomic distance in dataset
    # mu_max = np.max(D)
    # max_mol_size = Z.shape[1]

    # We predict values in the targets file
    # the targets file could be a txt file or an npz file.
    try:
        # Try loading the file as a txt file.
        y = np.loadtxt(path_to_targets_file).astype(dtype)
    except (ValueError,UnicodeDecodeError) as e:
        # Not a txt file, Try loading the file as an npz file.
        # UnicodeDecodeError in python3.6 NumPy (1.11.3)
        # ValueError in python2.7 NumPy (1.12.1)
        data_target = np.load(path_to_targets_file)
        assert len(data_target.files) == 1, "There appear to be more than one variable in the targets npz file: {}. There must be only one.".format(data_target.files)
        key = data_target.files[0]
        logger.info("Using the target {} from the targets npz file.".format(key))
        y = data_target[key].astype(dtype)

    values_to_predict = y.shape[1]

    # Split data for test and training
    Z_train, Z_test, D_train, D_test, y_train, y_test = train_test_split(
            Z, D, y, test_size=1.0-train_split, random_state=0)

    Z_test, Z_val, D_test, D_val, y_test, y_val = train_test_split(
            Z_test, D_test, y_test, test_size=valid_test_split, random_state=0)


    # idxs = range(y.shape[0])
    # idxs_train, idxs_test = train_test_split(idxs, test_size=1.0-train_split, random_state=0)
    # idxs_test, idxs_valid = train_test_split(idxs_test, test_size=valid_test_split, random_state=0)

    # pdb.set_trace()
    # # Splitting the data this way ensures we can work with pytables datastructures
    # # as well, which fail to split when passed directly to sklearn's train_test_plit
    # #
    # # NOTE : This still doesn't work for big data since the data is loaded to memory
    # # the moment we index into the pytable array this way.
    # X_train, X_val, X_test = X[idxs_train, :], X[idxs_valid, :], X[idxs_test, :]
    # y_train, y_val, y_test = y[idxs_train, :], y[idxs_valid, :], y[idxs_test, :]


    ### Following data splitting works only with numpy ndarrays and list
    ### doesn't work with pytables.

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1.0-train_split, random_state=0)

    X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=valid_test_split, random_state=0)

    # # print([len(_) for _ in (y_train,y_val,y_test)])
    # # Compute mean and standard deviation of per-atom-energy
    # Z_train_non_zero = np.count_nonzero(Z_train, axis=1)
    # Z_train_non_zero = np.expand_dims(Z_train_non_zero,axis=1)

    Estd = np.std(y_train, axis=0) # y values originally were free energies, they would be more when there are more atoms in the molecule, hence division scales them to be energy per atom.
    Emean = np.mean(y_train, axis=0) # axis needs to be specified so that we get mean and std per energy/spectrum value (i.e. dimension in y) doesn't affect when y just a scalar, i.e. free energy

    np.savez("X_vals.npz", X_train=X_train, X_test=X_test, X_val=X_val)
    np.savez("Y_vals.npz", Y_test=y_test, Y_train=y_train, Y_val=y_val, Y_mean=Emean, Y_std=Estd)

    return_dict = {
                "train_data": (X_train, y_train),
                "valid_data": (X_val, y_val),
                "test_data" : (X_test, y_test),
                "Estd":Estd, "Emean":Emean,
                "values_to_predict": values_to_predict,
                }

    return return_dict

def main(**params):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parent_dir = os.getcwd()
    mydir = os.path.join(os.getcwd(),timestamp)
    os.makedirs(mydir)
    os.chdir(mydir)
    os.mkdir("results")
    
    app_name="CNN"
    #global logger;
    # logger,fh,ch = get_logger(app_name=app_name, logfolder=mydir, fname="log_file_{}.log".format(timestamp))
    logger = get_logger(app_name=app_name, logfolder=mydir)
    logger.info("GOT LOGGER !")



    # save hyparameters
    np.savez("results/hyperparams.npz",  **params)
    data = get_data(logger=logger,**params)
    print("GOT DATA !")
    
    params_and_data = params.copy()
    params_and_data.update(data)
    
    error = train_and_get_error(logger=logger,**params_and_data)
    print("Test error :", error)
    
    #clean up
    os.chdir(parent_dir)
    return error

@click.command()
@click.option('--batch_size', default=10, help="The training batch size.")
@click.option('--model_name', default="orig_model", help="Name of the DTNN model")
@click.option('--learn_rate', type=click.FLOAT, default=None, help="The adam learning rate, if not set then a separate scheme is used.")
@click.option('--max_epochs', default=1000, help="Maximum number of training epochs.")
@click.option('--data_dim', default=30600, help="Length of the MBTR vector")
@click.option('--earlystop_epochs', default=100, help="If the validation error doesn't improve in these many epochs, the training is terminated.")
@click.option('--conv_filters', '-c', multiple=True, help="Number of convolution filters in respective layers, can be passed multiple times which corresponds to the layers in that order.")
@click.argument('path_to_coulomb_file')#,help="Path to the XYZ file.")
@click.argument('path_to_targets_file')#,help="Path to the energies or spectrum files.")
def get_params_and_goto_main(path_to_coulomb_file, path_to_targets_file, model_name, 
        data_dim, conv_filters, learn_rate=0.00001, earlystop_epochs=100,
        batch_size=100, trainSplit=0.9, valid_test_split=0.5, max_epochs=10000,
        cost="rmse"):
    
    params = {
            "path_to_coulomb_file":path_to_coulomb_file, 
            "path_to_targets_file":path_to_targets_file, 
            "train_split" : trainSplit,
            "valid_test_split" : valid_test_split,
            "max_epochs" :  max_epochs,
            "batch_size" : batch_size,
            "cost" : cost,
            "model_name" : model_name,
            "conv_filters" : [int(_) for _ in conv_filters],
            "learn_rate" : learn_rate,
            "earlystop_epochs" : earlystop_epochs,
            "data_dim" : data_dim # (1,coulomb_dim, coulomb_dim)
            }
    main(**params)



if __name__ == "__main__":
    get_params_and_goto_main() 
