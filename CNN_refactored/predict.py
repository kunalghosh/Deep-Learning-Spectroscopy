import gc
import os
import sys
import copy
from os import path
import pdb
# import cPickle as pickle
import pickle
import logging
import gzip
import timeit
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

def train_and_get_error(test_data, conv_filters, values_to_predict, batch_size, cost, 
        saved_params, model_name, logger, coulomb_dims, check_every=2,**kwargs):
    ## def train_and_get_error(train_data, valid_data, test_data, Estd, Emean, 
    ##        conv_filters, values_to_predict, max_epochs, batch_size, cost, 
    ##        model_name, learn_rate, earlystop_epochs, logger, coulomb_dims, 
    ##        check_every=2,**kwargs):    

    """
    param check_every      : Check the validation and test error every, these many epochs.
    param earlystop_epochs : If the validation error has not reduced in these many epochs
                             the training stops. It is decremented by 'check_every' depending
                             on the 'cost'. It is reset to its initial value if error reduces.
    """
    # X_train, y_train = train_data
    # X_val, y_val = valid_data
    # X_test, y_test    = test_data
    X_test = test_data

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
            outdim = values_to_predict, cost = cost, input_dims = coulomb_dims, **kwargs)
        
    lasagne.layers.set_all_param_values(l_out, saved_params)
    ## start_time = timeit.default_timer()

    ## lowest_test_mae = np.inf
    ## lowest_test_rmse = np.inf
    ## lowest_test_error = np.inf
    ## # mu_max = None#np.max(D_train)+1

    ## # D_val_fe = feature_expand(D_val, num_dist_basis, mu_max=mu_max)
    ## # D_test_fe = feature_expand(D_test, num_dist_basis,mu_max=mu_max)

    ## # logger.info("Expanded Test and Val.")

    ## num_train_samples = X_train.shape[0]
    ## num_train_batches = num_train_samples // batch_size

    ## earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)


    ## extend_dims = [-1]
    ## extend_dims.extend(coulomb_dims)
    ## X_train = X_train.reshape(-1, coulomb_dims[1], coulomb_dims[2])
    ## # pdb.set_trace()
    ## X_train_row_norms = np.linalg.norm(X_train, axis=1)

    for epoch in range(1):
        ## # Randomly shuffle coulomb matrix
        ## X_train = X_train.reshape(-1, coulomb_dims[1], coulomb_dims[2])
        ## X_train = coulomb_shuffle(X_train, X_train_row_norms)
        ## X_train = X_train.reshape(extend_dims)

        ## # Randomly permute training data
        ## rand_perm = rng.permutation(X_train.shape[0])
        ## X_train_perm = X_train[rand_perm]
        ## # D_train_perm = D_train[rand_perm]
        ## y_train_perm = y_train[rand_perm]

        ## # logger.info("Shuffled train data")
        ## if learn_rate is None:
        ##     if epoch < 10:
        ##     #if epoch < 50:
        ##         learning_rate = 0.01
        ##     elif epoch < 500:
        ##         learning_rate = 0.001
        ##     elif epoch < 5000:
        ##         learning_rate = 0.0001
        ##     else:
        ##         learning_rate = 0.00001
        ## else:
        ##     learning_rate = learn_rate

        ## train_cost = 0
        ## for batch in range(num_train_batches):
        ##     train_cost += f_train(
        ##             X_train_perm[batch*batch_size:((batch+1)*batch_size)],
        ##             # feature_expand(D_train_perm[batch*batch_size:((batch+1)*batch_size)], num_dist_basis, mu_max=mu_max),
        ##             y_train_perm[batch*batch_size:((batch+1)*batch_size)],
        ##             learning_rate
        ##             )
        ##     #print("miniBatch %d of %d done." % (batch, num_train_batches))
        ## train_cost = train_cost / num_train_batches
        ## # logger.info("Got training cost.")

        if (epoch % check_every) == 0:
            ## # y_pred = f_eval_test(Z_val, D_val_fe)
            ## y_pred = f_eval_test(X_val)
            ## val_errors = y_pred-y_val

            ## # logger.info("Got Y val_pred")
            logger.info("Trying to predict")
            y_pred = f_eval_test(X_test)
            logger.info("Got predictions")
            ## test_errors = y_pred-y_test
            ## test_cost = f_test(X_test, y_test)
            
            rmse = lambda x: np.sqrt(np.square(x).mean())
            mae  = lambda x: np.abs(x).mean()

            ## logger.info("VAL MAE:  %5.2f kcal/mol TEST MAE:  %5.2f kcal/mol" %
            ##         (mae(val_errors), mae(test_errors)))
            ## logger.info("VAL RMSE: %5.2f kcal/mol TEST RMSE: %5.2f kcal/mol" %
            ##         (rmse(val_errors), rmse(test_errors)))

            ## all_params = lasagne.layers.get_all_param_values(l_out)
            ## with gzip.open('results/model_epoch%d.pkl.gz' % (epoch), 'wb') as f:
            ##     pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)

            ## new_test_mae = mae(test_errors)
            ## if  new_test_mae < lowest_test_mae:
            ##     lowest_test_mae = new_test_mae
            ##     logger.info("Found best test MAE : {}".format(lowest_test_mae))
            ##     np.savez("Y_test_pred_best_mae.npz", Y_test_pred = y_pred)
            ##     if cost == "mae":
            ##         # Only update early stoppping counter if we are interested in this cost
            ##         earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)
            ##         lowest_test_error = lowest_test_mae

            ## new_test_rmse = rmse(test_errors)
            ## if new_test_rmse < lowest_test_rmse:
            ##     lowest_test_rmse = new_test_rmse
            ##     logger.info("Found best test RMSE : {}".format(lowest_test_rmse))
            ##     np.savez("Y_test_pred_best_rmse.npz", Y_test_pred = y_pred)
            ##     if cost == "rmse":
            ##         # Only update early stoppping counter if we are interested in this cost
            ##         earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)
            ##         lowest_test_error = lowest_test_rmse

            ## end_time = timeit.default_timer()

            ## test_error = np.sqrt(test_cost)
            ## train_error = np.sqrt(train_cost)
            ## logger.debug("Time %4.1f, Epoch %4d, train_cost=%5g, test_error=%5g" % (end_time - start_time, epoch, train_error, test_error))
            ## start_time = timeit.default_timer()

            ## # update and check the early stop counter.
            ## earlystop_epoch_counter -= check_every
            ## if earlystop_epoch_counter <= 0:
            ##     break

    # return test_error
    ## return lowest_test_error
    return y_pred

def get_data(path_to_coulomb_file, path_to_y_vals, path_to_saved_weights, logger,  
            coulomb_dims, dtype=theano.config.floatX, **kwargs):

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
        logger.info("Loading CM as txt")
        X = np.loadtxt(path_to_coulomb_file)
        logger.info("Shape of CM before reshaping : {}".format(X.shape))

    extend_dims = [-1]
    extend_dims.extend(coulomb_dims)
    X = X.reshape(extend_dims).astype(dtype)

    logger.info("X shape = {}".format(X.shape))

    # loading saved parameters
    with gzip.open(path_to_saved_weights, 'rb') as f:
        saved_params = pickle.load(f)

    ## We predict values in the targets file
    ## the targets file could be a txt file or an npz file.
    #try:
    #    # Try loading the file as a txt file.
    #    y = np.loadtxt(path_to_targets_file).astype(dtype)
    #except (ValueError,UnicodeDecodeError) as e:
    #    # Not a txt file, Try loading the file as an npz file.
    #    # UnicodeDecodeError in python3.6 NumPy (1.11.3)
    #    # ValueError in python2.7 NumPy (1.12.1)
    #    data_target = np.load(path_to_targets_file)
    #    assert len(data_target.files) == 1, "There appear to be more than one variable in the targets npz file: {}. There must be only one.".format(data_target.files)
    #    key = data_target.files[0]
    #    logger.info("Using the target {} from the targets npz file.".format(key))
    #    y = data_target[key].astype(dtype)

    Y_vals = np.load(path_to_y_vals)
    values_to_predict = Y_vals['Y_test'].shape[1]


    ## X_train, X_test, y_train, y_test = train_test_split(
    ##         X, y, test_size=1.0-train_split, random_state=0)

    ## X_test, X_val, y_test, y_val = train_test_split(
    ##         X_test, y_test, test_size=valid_test_split, random_state=0)
    ## # # print([len(_) for _ in (y_train,y_val,y_test)])
    ## # # Compute mean and standard deviation of per-atom-energy
    ## # Z_train_non_zero = np.count_nonzero(Z_train, axis=1)
    ## # Z_train_non_zero = np.expand_dims(Z_train_non_zero,axis=1)

    ## Estd = np.std(y_train, axis=0) # y values originally were free energies, they would be more when there are more atoms in the molecule, hence division scales them to be energy per atom.
    ## Emean = np.mean(y_train, axis=0) # axis needs to be specified so that we get mean and std per energy/spectrum value (i.e. dimension in y) doesn't affect when y just a scalar, i.e. free energy

    ## np.savez("X_vals.npz", X_train=X_train, X_test=X_test, X_val=X_val)
    ## np.savez("Y_vals.npz", Y_test=y_test, Y_train=y_train, Y_val=y_val, Y_mean=Emean, Y_std=Estd)

    return_dict = {
                # "train_data": (X_train, y_train),
                # "valid_data": (X_val, y_val),
                "test_data" : X,
                # "Estd":Estd, "Emean":Emean,
                "values_to_predict": values_to_predict,
                "saved_params" : saved_params,
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
    logger.info("Results for Saved weight at {}".format(params['path_to_saved_weights']))



    # save hyparameters
    np.savez("results/hyperparams.npz",  **params)
    data = get_data(logger=logger,**params)
    print("GOT DATA !")
    
    params_and_data = params.copy()
    params_and_data.update(data)
    
    y_pred = train_and_get_error(logger=logger,**params_and_data)
    np.savez("test_predictions.npz", y_pred=y_pred)
    ## print("Test error :", error)
    ## 
    ## #clean up
    ## os.chdir(parent_dir)
    ## return error

@click.command()
@click.option('--batch_size', default=100, help="The training batch size.")
@click.option('--model_name', default="orig_model", help="Name of the DTNN model")
# @click.option('--learn_rate', type=click.FLOAT, default=None, help="The adam learning rate, if not set then a separate scheme is used.")
# @click.option('--max_epochs', default=10000, help="Maximum number of training epochs.")
@click.option('--coulomb_dim', default=23, help="Dimensions of one side of a coulomb matrix (symmetric).")
# @click.option('--earlystop_epochs', default=100, help="If the validation error doesn't improve in these many epochs, the training is terminated.")
@click.option('--conv_filters', '-c', multiple=True, help="Number of convolution filters in respective layers, can be passed multiple times which corresponds to the layers in that order.")
@click.argument('path_to_coulomb_file')#,help="Path to the XYZ file.")
@click.argument('path_to_saved_weights')
@click.argument('path_to_y_vals') # to load the Training data's Emean and Estd
# @click.argument('path_to_targets_file')#,help="Path to the energies or spectrum files.")
def get_params_and_goto_main(path_to_coulomb_file, path_to_saved_weights, model_name, 
        path_to_y_vals, coulomb_dim, conv_filters, batch_size=100,cost="rmse"):
# def get_params_and_goto_main(path_to_coulomb_file, path_to_targets_file, model_name, 
#        coulomb_dim, conv_filters, learn_rate=0.00001, earlystop_epochs=100,
#        batch_size=100, trainSplit=0.9, valid_test_split=0.5, max_epochs=10000,
#        cost="rmse"):   
    params = {
            "path_to_coulomb_file":path_to_coulomb_file, 
            # "path_to_targets_file":path_to_targets_file, 
            "path_to_saved_weights":path_to_saved_weights, 
            # "train_split" : trainSplit,
            # "valid_test_split" : valid_test_split,
            # "max_epochs" :  max_epochs,
            "batch_size" : batch_size,
            "cost" : cost,
            "model_name" : model_name,
            "conv_filters" : [int(_) for _ in conv_filters],
            # "learn_rate" : learn_rate,
            # "earlystop_epochs" : earlystop_epochs,
            "coulomb_dims" : (1,coulomb_dim, coulomb_dim),
            "path_to_y_vals" : path_to_y_vals
            }
    main(**params)



if __name__ == "__main__":
    get_params_and_goto_main() 
