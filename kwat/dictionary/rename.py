from numpy import full, sort, unique


def rename(na_, na_re, ke=True):

    n_na = len(na_)

    fa_ = []

    re_ = full(n_na, "", dtype=object)

    for ie, na in enumerate(na_):

        if na in na_re:

            re = na_re[na]

        else:

            fa_.append(na)

            if ke:

                re = na

            else:

                re = None

        re_[ie] = re

    n_su = n_na - len(fa_)

    fa_ = sort(unique(fa_))

    print("Renamed {} ({:.2%}) (failed {})".format(n_su, n_su / n_na, fa_.size))

    return re_, fa_
