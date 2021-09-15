from numpy import array, full, isnan, triu_indices
from scipy.spatial.distance import squareform


def get_coclustering_distance(gr_po_tr):

    pa_ = array(triu_indices(gr_po_tr.shape[0], k=1)).T

    n_pa = len(pa_)

    di_ = full(n_pa, 0)

    n_to = gr_po_tr.shape[1]

    for iep in range(n_pa):

        gr_pop_tr = gr_po_tr[pa_[iep]]

        n_tr = 0

        n_co = 0

        for iet in range(n_to):

            gr1, gr2 = gr_pop_tr[:, iet]

            if not (isnan(gr1) or isnan(gr2)):

                n_tr += 1

                n_co += int(gr1 == gr2)

        di_[iep] = 1 - n_co / n_tr

    return squareform(di_)
