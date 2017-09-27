import pprint
import numpy as np
from numpy.random import seed
import GPy
import GPyOpt
# from train import main
import matplotlib
matplotlib.use('Agg')
import pickle as pkl

# data_path = "/m/home/home0/00/ghoshk1/data/Desktop/Thesis/data/Annika_new/132k_16_opt_eV"
# results_pkl = "/u/00/ghoshk1/unix/Desktop/Thesis/code/thesis_code/MLP_refactored/results_2-7.pkl"
# path_to_coulomb_file = data_path + "/coulomb.npz"
# path_to_targets_file = data_path + "/energies.txt"
# max_epochs = 500
# model_name = "orig_model"
# coulomb_dim = 29
# activation = "rectify"

# The Montavon MLP was used and trained for 200,000 epochs
# Hyperparameters optimized:
# Minibatch size.
# Number of units in the two layers.

# NOTE THAT THIS OPTIMIZATION WAS RUN MANUALLY !!!!
def optimize():
    bounds = [
            #{'name':'learn_rate', 'type': 'discrete', 'domain':range(2,7)}, # 5 
            {'name':'batch_size', 'type':'discrete', 'domain': np.arange(20, 201, 5)},  # 37
            {'name':'num_units1', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            {'name':'num_units2', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units3', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units4', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units5', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units6', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units7', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units8', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units9', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            # {'name':'num_units10', 'type':'discrete', 'domain': np.arange(100, 600, 50)}, # 12
            ]

    # try:
    #     with open(results_pkl, "rb") as f:
    #         results = pkl.load(f)
    #         print("Loaded {}".format(results_pkl))
    # except Exception as e:
    #     print(e)
    #     print("Couldn't open {}, creating new one.".format(results_pkl))
    #     results = {}
    results = {(80, 550, 150): 0.30403,
            (190, 100, 250) : 100, # run failed
            (35, 250, 150) : 0.32642,
            (75, 200, 250) : 0.31581,
            (20, 250, 450) : 0.33723,
            (85, 550, 150) : 100,
            (75, 550, 150) : 0.30604,
            (70, 550, 150) : 0.30782,
            (65, 550, 150) : 0.30839,
            (60, 550, 150) : 0.31195,
            (70, 200, 250) : 0.31747,
            (30, 250, 150) : 0.33149,
            (25, 250, 450) : 0.33429,
            (55, 550, 150) : 0.31486,
            (45, 550, 150) : 0.31886}  

    def wrapper_function(gpy_input):
        gpy_inputs = np.int32(gpy_input[0])
        inputs_tuple = tuple(gpy_inputs) 
        print("Params : {}".format(inputs_tuple))
        if inputs_tuple in results.keys():
            retval = results[inputs_tuple]
            print("Returning precomputed results : {}".format(retval))
            return retval
        else:
            print("Run the experiment and update the dictionary. Exitting now..")
            exit(1)

        # learn_rate, batch_size, num_units = gpy_inputs[0], gpy_inputs[1], gpy_inputs[2:]
        # learn_rate = np.float32(10.0**(-learn_rate))

        # params = {
        # "path_to_coulomb_file":path_to_coulomb_file, 
        # "path_to_targets_file":path_to_targets_file, 
        # "train_split" : 0.9,
        # "valid_test_split" : 0.5,
        # "max_epochs" :  max_epochs,
        # "batch_size" : batch_size,
        # "cost" : "rmse",
        # "model_name" : model_name,
        # "units_list" : num_units,
        # "learn_rate" : learn_rate,
        # "earlystop_epochs" : 100,
        # "coulomb_dims" : (1,coulomb_dim, coulomb_dim),
        # "activation" : activation
        # }
        # pprint.pprint(params)
        # error = main(**params)

        # results[inputs_tuple] = error

        # return error 

    bayes_opt = GPyOpt.methods.BayesianOptimization(wrapper_function,
            domain = bounds,
            acquisition_type='EI',
            exact_feval=False
            )
    bayes_opt.run_optimization(max_iter=10)
    print("Suggested sample: {}".format(bayes_opt.suggested_sample))

    # print("Params and error:\n")
    # pprint.pprint(results)

    # print("Writing params and corresponding errors to {}.".format(results))
    # with open(results_pkl, "wb") as f:
    #     pkl.dump(results, f)

    bayes_opt.plot_acquisition("acquisition_200epoch.png")
    bayes_opt.plot_convergence("convergence_200epoch.png")

if __name__ == "__main__":
    seed(1234)
    optimize()

