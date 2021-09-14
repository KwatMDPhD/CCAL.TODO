from multiprocessing import Pool

from numpy import array

from ..array_array import apply, separate_and_apply


def apply_with_vector(ta, ro_, fu, se=False, n_jo=1):

    if se:

        ap = separate_and_apply

    else:

        ap = apply

    po = Pool(processes=n_jo)

    an_ = array(po.starmap(ap, ([ta, ro, fu] for ro in ro_)))

    po.terminate()

    return an_
