from multiprocessing import Pool

from numpy import array, full, nan

from .function import ignore_nan_and_apply, separate_and_apply


def compare_with_target(ta, ro_, fu, separate=False, n_jo=1):

    if separate:

        ap = separate_and_apply

    else:

        ap = ignore_nan_and_apply

    po = Pool(n_jo)

    re_ = array(po.starmap(ap, ((ro, ta, fu) for ro in ro_)))

    po.terminate()

    return re_


def compare_with_other(ro1_, ro2_, fu):

    n_ro1 = ro1_.shape[0]

    n_ro2 = ro2_.shape[0]

    re_ = full((n_ro1, n_ro2), nan)

    for ie1 in range(n_ro1):

        ro1 = ro1_[ie1]

        for ie2 in range(n_ro2):

            ro2 = ro2_[ie2]

            re_[ie1, ie2] = fu(ro1, ro2)

    return re_
