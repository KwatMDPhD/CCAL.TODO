def reflect(co_, re):
    cor_ = co_.copy()

    for ie, co in enumerate(co_):
        if co < re:
            cor_[ie] += (re - co) * 2

        else:
            cor_[ie] -= (co - re) * 2

    return cor_
