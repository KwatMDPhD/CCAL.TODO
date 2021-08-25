from numpy import full, nan


def compare_with_other(ro1_, ro2_, fu):

    n_ro1 = ro1_.shape[0]

    n_ro2 = ro2_.shape[0]

    fu_ro1_ro2 = full([n_ro1, n_ro2], nan)

    for ie1 in range(n_ro1):

        ro1 = ro1_[ie1]

        for ie2 in range(n_ro2):

            ro2 = ro2_[ie2]

            fu_ro1_ro2[ie1, ie2] = fu(ro1, ro2)

    return fu_ro1_ro2
