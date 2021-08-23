from numpy import array, full, isnan, triu_indices
from scipy.spatial.distance import squareform


def _get_coclustering_distance(cl_po_tr):

    pa_ = array(triu_indices(cl_po_tr.shape[0], k=1)).T

    n_pa = pa_.shape[0]

    di_ = full(n_pa, 0)

    n_cl = cl_po_tr.shape[1]

    for iepa in range(n_pa):

        cl_popa_cl = cl_po_tr[pa_[iepa]]

        n_tr = 0

        n_co = 0

        for iecl in range(n_cl):

            cl1, cl2 = cl_popa_cl[:, iecl]

            if not (isnan(cl1) or isnan(cl2)):

                n_tr += 1

                n_co += int(cl1 == cl2)

        di_[iepa] = 1 - n_co / n_tr

    return squareform(di_)
