from numpy import unique

from ..constant import NUMBER_OF_CATEGORY


def guess_type(nu___, n_ca=NUMBER_OF_CATEGORY):

    if all(float(nu).is_integer() for nu in nu___.ravel()):

        n_un = unique(nu___).size

        if n_un <= 2:

            return "binary"

        elif n_un <= n_ca:

            return "categorical"

    else:

        return "continuous"
