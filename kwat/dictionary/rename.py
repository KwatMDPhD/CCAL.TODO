from numpy import full, sort, unique


def rename(na_, na_re, ke=True):

    n_na = len(na_)

    re_ = full(n_na, "", dtype=object)

    n_su = 0

    fa_ = []

    for ie, na in enumerate(na_):

        if na in na_re:

            n_su += 1

            re = na_re[na]

        else:

            fa_.append(na)

            if ke:

                re = na

            else:

                re = None

        re_[ie] = re

    fa_ = sort(unique(fa_))

    n_fa = fa_.size

    print(
        "Renamed {} ({:.2%}) failed {} ({:.2%})".format(
            n_su, n_su / n_na, n_fa, n_fa / n_na
        )
    )

    return re_, fa_
