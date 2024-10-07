from numpy.random import choice, seed

from ..constant import RANDOM_SEED


def sample(da, sh, ra=RANDOM_SEED, **ke_ar):
    n_ro, n_co = da.shape

    n_ros, n_cos = sh

    seed(seed=ra)

    if n_ros is not None:
        if n_ros < 1:
            n_ros = int(n_ro * n_ros)

        da = da.iloc[choice(n_ro, size=n_ros, **ke_ar), :]

    if n_cos is not None:
        if n_cos < 1:
            n_cos = int(n_co * n_cos)

        da = da.iloc[:, choice(n_co, size=n_cos, **ke_ar)]

    return da
