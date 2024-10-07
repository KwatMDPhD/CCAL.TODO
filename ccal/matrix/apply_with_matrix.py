from numpy import full, nan


def apply_with_matrix(ma1, ma2, fu):
    n_ro1 = ma1.shape[0]

    n_ro2 = ma2.shape[0]

    fu_ro1_ro2 = full([n_ro1, n_ro2], nan)

    for ie1 in range(n_ro1):
        ro1 = ma1[ie1]

        for ie2 in range(n_ro2):
            fu_ro1_ro2[ie1, ie2] = fu(ro1, ma2[ie2])

    return fu_ro1_ro2
