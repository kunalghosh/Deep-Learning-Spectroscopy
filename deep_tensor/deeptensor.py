import sys
import cPickle as pickle
import gzip
import timeit
import numpy as np
import lasagne
import theano.tensor as T
import theano
from utils import load_qm7b_data, load_oqmd_data
from sklearn.model_selection import train_test_split

theano.config.floatX = 'float32'

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

    # Load data
    Z, D, y, num_species = load_qm7b_data(num_dist_basis, dtype=theano.config.floatX)
    #Z, D, y, num_species = load_oqmd_data(num_dist_basis, dtype=theano.config.floatX, filter_query="natoms<10,computation=standard")
    max_mol_size = Z.shape[1]

    # Split data for test and training
    Z_train, Z_test, D_train, D_test, y_train, y_test = train_test_split(
            Z, D, y, test_size=0.2, random_state=0)

    # Compute mean and standard deviation of per-atom-energy
    Estd = np.std(y_train / np.count_nonzero(Z_train, axis=1))
    Emean = np.mean(y_train / np.count_nonzero(Z_train, axis=1))

    sym_Z = T.imatrix()
    sym_D = T.tensor4()
    sym_y = T.vector()
    sym_learn_rate = T.scalar()

    l_in_Z = lasagne.layers.InputLayer((None, max_mol_size))
    l_in_D = lasagne.layers.InputLayer((None, max_mol_size, max_mol_size, num_dist_basis))
    l_mask = MaskLayer(l_in_Z)
    l_c0 = SwitchLayer(l_in_Z, num_species, c_len, W=lasagne.init.Uniform(1.0/np.sqrt(c_len)))

    l_cT = RecurrentLayer(l_c0, l_in_D, l_mask, num_passes=num_interaction_passes, num_hidden=num_hidden_neurons)

    # Compute energy contribution from each atom
    l_atom1 = lasagne.layers.DenseLayer(l_cT, 15, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2)
    l_atom2 = lasagne.layers.DenseLayer(l_atom1, 1, nonlinearity=None, num_leading_axes=2)
    l_atom2 = lasagne.layers.FlattenLayer(l_atom2, outdim=2) # Flatten singleton dimension
    l_atomE = lasagne.layers.ExpressionLayer(l_atom2, lambda x: (x*Estd+Emean)) # Scale and shift by mean and std deviation
    l_out = SumMaskedLayer(l_atomE, l_mask)

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        print("%s, %s" % (p, p.get_value().shape))

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
            print("TRAIN MAE:  %5.2f kcal/mol TEST MAE:  %5.2f kcal/mol" %
                    (np.abs(train_errors).mean(), np.abs(test_errors).mean()))
            print("TRAIN RMSE: %5.2f kcal/mol TEST RMSE: %5.2f kcal/mol" %
                    (np.sqrt(np.square(train_errors).mean()), np.sqrt(np.square(test_errors).mean())))

            all_params = lasagne.layers.get_all_param_values(l_out)
            with gzip.open('results/model_epoch%d.pkl.gz' % (epoch), 'wb') as f:
                pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)

        if (epoch % 2) == 0:
            test_cost = f_test(Z_test, D_test, y_test)
            end_time = timeit.default_timer()

            print("Time %4.1f, Epoch %4d, train_cost=%5g, test_error=%5g" % (end_time - start_time, epoch, np.sqrt(train_cost), np.sqrt(test_cost)))
            start_time = timeit.default_timer()

if __name__ == "__main__":
    main()
