def reflect_1d_grid(co_, re):

    co2_ = co_.copy()

    for ie, co in enumerate(co_):

        if co < re:

            co2_[ie] += (re - co) * 2

        else:

            co2_[ie] -= (co - re) * 2

    return co2_
