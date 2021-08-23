from ._map_broad import _map_broad


def rename(na_):

    na_re = _map_broad()

    re_ = []

    fa_ = []

    for na in na_:

        re = na_re.get(na.lower())

        re_.append(re)

        if re is None:

            fa_.append(na)

    if 0 < len(fa_):

        print("Failed {}.".format(sorted(set(fa_))))

    return re_
