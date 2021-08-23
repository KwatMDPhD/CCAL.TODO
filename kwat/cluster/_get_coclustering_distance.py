from numpy import array, full, isnan, triu_indices
from scipy.spatial.distance import squareform


def _get_coclustering_distance(cl_po_tr):

    pa_ = array(triu_indices(cl_po_tr.shape[0], k=1)).T

    n_pa = pa_.shape[0]

    di_ = full(n_pa, 0)

    n_tr = cl_po_tr.shape[1]

    for iep in range(n_pa):

        cl_pop_tr = cl_po_tr[pa_[iep]]

        n_tr = 0

        n_co = 0

        for iet in range(n_tr):

            cl1, cl2 = cl_pop_tr[:, iet]

            if not (isnan(cl1) or isnan(cl2)):

                n_tr += 1

                n_co += int(cl1 == cl2)

        di_[iep] = 1 - n_co / n_tr

    return squareform(di_)
