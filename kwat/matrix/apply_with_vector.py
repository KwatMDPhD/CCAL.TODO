from multiprocessing import Pool

from numpy import array

from ..array_array import apply, separate_and_apply


def apply_with_vector(ve, ma, fu, se=False, n_jo=1):

    if se:

        ap = separate_and_apply

    else:

        ap = apply

    po = Pool(processes=n_jo)

    re_ = array(po.starmap(ap, ([ve, ro, fu] for ro in ma)))

    po.terminate()

    return re_
