import os
import sys
from os import path
import pdb
import cPickle as pickle
import gzip
import timeit
import numpy as np
import lasagne
import theano.tensor as T
import theano
from utils import load_qm7b_data, load_oqmd_data
from sklearn.model_selection import train_test_split
from datetime import datetime
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from util import get_logger

theano.config.floatX = 'float32'


mydir = os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)
os.chdir(mydir)
os.mkdir("results")

app_name = sys.argv[0]
logger = get_logger(app_name=app_name, logfolder=mydir)

path_to_xyz_file = sys.argv[1]

# The following targets file could be
# 16 properties or 300 values of the spectrum.
# The neural network output dimensions are determined from the
# second dimension of the targets tensor once loaded. 
# (The first dimension is the datapoints.)

try:
    path_to_targets_file = sys.argv[2]
    logger.info("Predicting targets {}".format(path_to_targets_file))
except IndexError as e :
    # Predict free energies instead.
    path_to_targets_file = None
    logger.info("Predicting free energies [No targets file provided].")

remove5koutliers = False
try:
    if "remove5koutliers" in sys.argv[3:]:
        remove5koutliers = True
        logger.info("REMOVING 5k outliers.")
except IndexError as e :
    pass
finally:
    if remove5koutliers is False:
        logger.info("NOT Removing 5k outliers.")

class SwitchLayer(lasagne.layers.Layer):
    """
    Layer contains a coefficient matrix.
    Rows from this matrix are returned using the input as row indices.
    The output array thus has an additional dimension.

    Parameters
    ----------
    incoming : :class: `Layer` instances

    num_options : int or T.scalar
        Number of rows in the coefficient matrix

    out_len : int or T.scalar
        Number of columns in coefficient matrix
    """
    def __init__(self, incoming, num_options, out_len, W=lasagne.init.Uniform(1), **kwargs):
        super(SwitchLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.out_len = out_len
        self.C = self.add_param(W, (num_options, self.out_len), name='C')

    def get_output_for(self, input, **kwargs):
        return self.C[input,:]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_len)

class MaskLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input>0

class SumMaskedLayer(lasagne.layers.MergeLayer):
    def __init__(self, var, mask, **kwargs):
        super(SumMaskedLayer, self).__init__([var, mask], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0]

    def get_output_for(self, input, **kwargs):
        var, mask = input
        return T.sum(var*mask, axis=1)

class RecurrentLayer(lasagne.layers.MergeLayer):
    """
        Layer implements the iterative refinement of atom coefficients as
        described in [SCHUTT].

        References
        ----------
        [SCHUTT] Schutt, Kristof T., Farhad Arbabzadah, Stefan Chmiela, Klaus
        R. Muller, and Alexandre Tkatchenko. 2017. "Quantum-Chemical Insights
        from Deep Tensor Neural Networks." Nature Communications 8 (January):
        13890.
    """
    def __init__(self, atomc, dist, atom_mask, num_hidden=60, num_passes=2, include_diagonal=False, nonlinearity=lasagne.nonlinearities.tanh, Wcf=lasagne.init.GlorotNormal(1.0), Wfc=lasagne.init.GlorotNormal(1.0), Wdf=lasagne.init.GlorotNormal(1.0), bcf=lasagne.init.Constant(0.0), bdf=lasagne.init.Constant(0.0), **kwargs):
        super(RecurrentLayer, self).__init__([atomc, dist, atom_mask], **kwargs)
        num_atoms = self.input_shapes[0][1]
        c_len = self.input_shapes[0][2]
        d_len = self.input_shapes[1][3]
        self.Wcf = self.add_param(Wcf, (c_len, num_hidden), name="W_atom_c")
        self.bcf = self.add_param(bcf, (num_hidden, ), name="b_atom_c")
        self.Wdf = self.add_param(Wdf, (d_len, num_hidden), name="W_dist")
        self.bdf = self.add_param(bdf, (num_hidden, ), name="b_dist")
        self.Wfc = self.add_param(Wfc, (num_hidden, c_len), name="W_hidden_to_c")
        self.num_passes = num_passes
        self.nonlin = nonlinearity
        if include_diagonal:
            self.inv_eye_mask = None
        else:
            self.inv_eye_mask = (T.eye(num_atoms,num_atoms) < 1).dimshuffle("x",0,1,"x")


    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        atom_c, dist, atom_mask = input

        c = atom_c

        for i in range(self.num_passes):
            # Contribution from atoms C
            # c has dim (sample, atom_i, feature)
            #   dimshuffle makes it broadcastable as (sample, 1, atom_j, feature)
            term1 = (T.dot(c.dimshuffle(0,"x",1,2), self.Wcf) + self.bcf)
            # Contribution from distances
            # dist has dim (sample, atom_i, atom_j, gaussian_expansion)
            term2 = T.dot(dist, self.Wdf) + self.bdf
            V = self.nonlin(T.dot(term1*term2, self.Wfc))
            # V has dim (sample, atom_i, atom_j, feature)
            # Atom mask zeroes out contribution from missing atoms
            # inv_eye_mask zeroes out contribution from diagonal V_{i,i}
            if self.inv_eye_mask is None:
                masked_V = V*atom_mask.dimshuffle(0,"x",1,"x")
            else:
                masked_V = V*atom_mask.dimshuffle(0,"x",1,"x")*self.inv_eye_mask
            Vsum = T.sum(masked_V, axis=2)
            # Vsum has dim (sample, atom_i, feature)
            c = c + Vsum

        return c


def main():
    rng = np.random.RandomState(4)
    np.random.seed(1)

    # Define model parameters
    num_dist_basis = 40
    c_len = 30
    num_hidden_neurons = 60
    num_interaction_passes = 2
    values_to_predict = 1

    # Load data
    Z, D, y, num_species = load_qm7b_data(num_dist_basis, dtype=theano.config.floatX, xyz_file=path_to_xyz_file)
    #Z, D, y, num_species = load_oqmd_data(num_dist_basis, dtype=theano.config.floatX, filter_query="natoms<10,computation=standard")
    max_mol_size = Z.shape[1]

    if path_to_targets_file is not None:
        # We predict values in the targets file
        y = np.loadtxt(path_to_targets_file).astype(np.float32)
        values_to_predict = y.shape[1]

    if remove5koutliers:
        assert y.shape[1] == 16, "Y.shape[1] != 16. Remove 5k outliers only useful for energy files."
        from get_idxs_to_keep import get_idxs_to_keep
        idxs = get_idxs_to_keep(path_to_targets_file)
        Z = Z[idxs,:]
        D = D[idxs,:]
        y = y[idxs,:]

    # Split data for test and training
    Z_train, Z_test, D_train, D_test, y_train, y_test = train_test_split(
            Z, D, y, test_size=0.2, random_state=0)

    # Compute mean and standard deviation of per-atom-energy
    Z_train_non_zero = np.count_nonzero(Z_train, axis=1)

    if path_to_targets_file is not None:
        Z_train_non_zero = np.expand_dims(Z_train_non_zero,axis=1)

    Estd = np.std(y_train / Z_train_non_zero, axis=0) # y values originally were free energies, they would be more when there are more atoms in the molecule, hence division scales them to be energy per atom.
    Emean = np.mean(y_train / Z_train_non_zero, axis=0) # axis needs to be specified so that we get mean and std per energy/spectrum value (i.e. dimension in y) doesn't affect when y just a scalar, i.e. free energy

    np.savez("X_vals.npz", Z_train=Z_train, Z_test=Z_test, D_train=D_train, D_test=D_test)
    np.savez("Y_vals.npz", Y_test=y_test, Y_train=y_train, Y_mean=Emean, Y_std=Estd)

    sym_Z = T.imatrix()
    sym_D = T.tensor4()
    sym_y = T.vector()
    if path_to_targets_file is not None:
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
    if path_to_targets_file is None:
        l_atom2 = lasagne.layers.FlattenLayer(l_atom2, outdim=2) # Flatten singleton dimension # outdim (-1, 23)
        # but if path_to_targets is not None, then we don't want to flatten since we want to get outputs (energies, or spectrum values) for each atom: ie. we want outdim (-1, 23, values_to_predict)
    l_atomE = lasagne.layers.ExpressionLayer(l_atom2, lambda x: (x*Estd+Emean)) # Scale and shift by mean and std deviation
    if path_to_targets_file is not None:
        l_mask = lasagne.layers.ReshapeLayer(l_mask, ([0], [1], 1)) # add an extra dimension so that l_atomE (-1, 23, 16) l_mask "after reshape" (-1, 23, 1) can be multiplied
    l_out = SumMaskedLayer(l_atomE, l_mask) # TODO : BUG HERE.

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        logger.debug("%s, %s" % (p, p.get_value().shape))

    out_train = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=False)
    out_test = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=True)
    cost_train = T.mean(lasagne.objectives.squared_error(out_train, sym_y))
    cost_test = T.mean(lasagne.objectives.squared_error(out_test, sym_y))
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

    # Define training parameters
    batch_size = 64
    num_train_samples = Z_train.shape[0]
    num_train_batches = num_train_samples // batch_size
    max_epochs=10000
    start_time = timeit.default_timer()

    lowest_test_mae = np.inf
    lowest_test_rmse = np.inf

    for epoch in range(max_epochs):
        # Randomly permute training data
        rand_perm = rng.permutation(Z_train.shape[0])
        Z_train_perm = Z_train[rand_perm]
        D_train_perm = D_train[rand_perm]
        y_train_perm = y_train[rand_perm]

        if epoch < 50:
            learning_rate = 0.01
        elif epoch < 500:
            learning_rate = 0.001
        elif epoch < 3000:
            learning_rate = 0.0001
        else:
            learning_rate = 0.00001

        train_cost = 0
        for batch in range(num_train_batches):
            train_cost += f_train(
                    Z_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    D_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    y_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    learning_rate
                    )
        train_cost = train_cost / num_train_batches

        if (epoch % 30) == 0:
            y_pred = f_eval_test(Z_train, D_train)
            train_errors = y_pred-y_train
            y_pred = f_eval_test(Z_test, D_test)
            test_errors = y_pred-y_test
            logger.info("TRAIN MAE:  %5.2f kcal/mol TEST MAE:  %5.2f kcal/mol" %
                    (np.abs(train_errors).mean(), np.abs(test_errors).mean()))
            logger.info("TRAIN RMSE: %5.2f kcal/mol TEST RMSE: %5.2f kcal/mol" %
                    (np.sqrt(np.square(train_errors).mean()), np.sqrt(np.square(test_errors).mean())))

            all_params = lasagne.layers.get_all_param_values(l_out)
            with gzip.open('results/model_epoch%d.pkl.gz' % (epoch), 'wb') as f:
                pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)

            new_test_mae = np.abs(test_errors).mean()
            if  new_test_mae < lowest_test_mae:
                lowest_test_mae = new_test_mae
                logger.info("Found best test MAE : {}".format(lowest_test_mae))
                np.savez("Y_test_pred_best_mae.npz", Y_test_pred = y_pred)

            new_test_rmse = np.sqrt(np.square(test_errors).mean())
            if new_test_rmse < lowest_test_rmse:
                lowest_test_rmse = new_test_rmse
                logger.info("Found best test RMSE : {}".format(lowest_test_rmse))
                np.savez("Y_test_pred_best_rmse.npz", Y_test_pred = y_pred)

        if (epoch % 2) == 0:
            test_cost = f_test(Z_test, D_test, y_test)
            end_time = timeit.default_timer()

            logger.debug("Time %4.1f, Epoch %4d, train_cost=%5g, test_error=%5g" % (end_time - start_time, epoch, np.sqrt(train_cost), np.sqrt(test_cost)))
            start_time = timeit.default_timer()

    print("Execution complete. Save the Y values")

if __name__ == "__main__":
    main()
