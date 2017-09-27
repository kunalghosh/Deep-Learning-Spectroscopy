import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
import theano

class NaiveCoulombShuffleLayer(lasagne.layers.Layer):
    """
    Assumes the input to be a minibatch of coulomb matrices [BATCH, 1, 29, 29] 
    The shuffling is as described in [MONTAVON]_.

    Parameters
    ----------
    coulomb: :class:`Layer` instances
        Parameterizing the coulomb matrix as described in 
        [MONTAVON]_. The code assumes that these have the
        same number of dimensions.

    seed : int
        seed to random stream

    axis : int
        the dimension to permute

    Methods
    ----------
    seed : Helper function to change the random seed after init is called

    References
    ----------
        ..  [MONTAVON] Grégoire Montavon et al 2013 New J. Phys. 15 095003 
            "Machine learning of molecular electronic properties in chemical compound space"
            http://iopscience.iop.org/article/10.1088/1367-2630/15/9/095003#citations.
    """
    def __init__(self, coulomb,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 axis=2,
                 **kwargs):
        super(NaiveCoulombShuffleLayer, self).__init__([coulomb], **kwargs)
        self._srng = RandomStreams(seed)
        self.axis = axis

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
       self._srng.seed(seed)

    def get_output_shape_for(self, input_shapes):
        #The shape doesn't change, this layer only shuffles the rows.
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        coulomb = input
        shape = coulomb.shape
        # coulomb matrices are symmetric so it doesn't matter which axis we find the norm over
        norm = T.norm(coulomb, axis = self.axis) 
        # we want as many random numbers as (batchsize, axis representing coulomb matrix rows) 
        eps = self._srng.normal(norm.shape)
        idxs = T.argsort(norm + eps)
        z = mu + T.exp(0.5 * log_var) * eps
        return z

