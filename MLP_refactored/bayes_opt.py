import pprint
import numpy as np
from numpy.random import seed
import GPy
import GPyOpt
from train import main
import matplotlib
matplotlib.use('Agg')

data_path = "/m/home/home0/00/ghoshk1/data/Desktop/Thesis/data/Annika_new/132k_16_opt_eV"
path_to_coulomb_file = data_path + "/coulomb.npz"
path_to_targets_file = data_path + "/energies.txt"
max_epochs = 500
model_name = "orig_model"
coulomb_dim = 29

def optimize():
    bounds = [
            {'name':'learn_rate', 'type': 'discrete', 'domain':range(2,7)}, # 5 
            {'name':'batch_size', 'type':'discrete', 'domain': np.arange(20, 201, 5)},  # 37
            {'name':'num_filters1', 'type':'discrete', 'domain': np.arange(2, 60, 5)}, # 12
            {'name':'num_filters2', 'type':'discrete', 'domain': np.arange(2, 60, 5)}, # 12
            {'name':'num_filters3', 'type':'discrete', 'domain': np.arange(2, 60, 5)}, # 12
            ]

    def wrapper_function(gpy_input):
        print("Params : {}".format(gpy_input))
        gpy_inputs = np.int32(gpy_input[0])
        learn_rate, batch_size, num_filters = gpy_inputs[0], gpy_inputs[1], gpy_inputs[2:]
        learn_rate = np.float32(10.0**(-learn_rate))

        params = {
        "path_to_coulomb_file":path_to_coulomb_file, 
        "path_to_targets_file":path_to_targets_file, 
        "train_split" : 0.9,
        "valid_test_split" : 0.5,
        "max_epochs" :  max_epochs,
        "batch_size" : batch_size,
        "cost" : "rmse",
        "model_name" : model_name,
        "conv_filters" : num_filters,
        "learn_rate" : learn_rate,
        "earlystop_epochs" : 100,
        "coulomb_dims" : (1,coulomb_dim, coulomb_dim)
        }
        pprint.pprint(params)
        error = main(**params)
        return error 

    bayes_opt = GPyOpt.methods.BayesianOptimization(wrapper_function,
            domain = bounds,
            acquisition_type='EI',
            exact_feval=False
            )
    bayes_opt.run_optimization(max_iter=10)
    print("Suggested sample: {}".format(bayes_opt.suggested_sample))
    bayes_opt.plot_acquisition("acquisition_200epoch.png")
    bayes_opt.plot_convergence("convergence_200epoch.png")

if __name__ == "__main__":
    seed(1234)
    optimize()
