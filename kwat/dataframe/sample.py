from numpy.random import choice, seed

from ..constant import RANDOM_SEED


def sample(da, sh, ra=RANDOM_SEED, **ke):

    si1, si2 = da.shape

    sa1, sa2 = sh

    seed(ra)

    if sa1 is not None:

        if sa1 < 1:

            sa1 = int(sa1 * si1)

        da = da.iloc[choice(si1, sa1, **ke), :]

    if sa2 is not None:

        if sa2 < 1:

            sa2 = int(sa2 * si2)

        da = da.iloc[:, choice(si2, sa2, **ke)]

    return da
