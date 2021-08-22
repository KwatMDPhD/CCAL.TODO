from ._map_broad import _map_broad


def rename(na_):

    na_ce = _map_broad()

    ce_ = []

    fa_ = []

    for na in na_:

        nalo = na.lower()

        if nalo in na_ce:

            ce_.append(na_ce[nalo])

        else:

            ce_.append(None)

            fa_.append(na)

    if 0 < len(fa_):

        print("Failed {}.".format(sorted(set(fa_))))

    return ce_
