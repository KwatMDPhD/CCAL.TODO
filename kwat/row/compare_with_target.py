from multiprocessing import Pool

from numpy import array

from ..function import ignore_nan_and_apply, separate_and_apply


def compare_with_target(ta, ro_, fu, separate=False, n_jo=1):

    if separate:

        ap = separate_and_apply

    else:

        ap = ignore_nan_and_apply

    po = Pool(processes=n_jo)

    re_ = array(po.starmap(ap, ([ro, ta, fu] for ro in ro_)))

    po.terminate()

    return re_
