import pdb
import numpy as np

def coulomb_shuffle(coulomb_matrices, row_norms):
    """
    Given coulomb matrices [batch, 29, 29] and its corresponding
    row_norms, this function shuffles the rows as described in [MONTAVON]_.

    Input is assumed to have a shape [batch, width, height] where width == height

    References
    ----------
        ..  [MONTAVON] Grégoire Montavon and Matthias Rupp and Vivekanand Gobre and Alvaro Vazquez-Mayagoitia and Katja Hansen and Alexandre Tkatchenko and Klaus-Robert Müller and O Anatole von Lilienfeld.
            "Machine learning of molecular electronic properties in chemical compound space}"
            New Journal of Physics http://stacks.iop.org/1367-2630/15/i=9/a=095003 (2013).
    """
    row_norms = row_norms + np.random.normal(size=row_norms.shape)
    sortidxs = np.argsort(row_norms)[:,::-1]
    #pdb.set_trace()
    coulomb_matrices = np.array([coulomb_matrices[_][sortidxs[_]] for _ in range(coulomb_matrices.shape[0])]) 
    return coulomb_matrices
