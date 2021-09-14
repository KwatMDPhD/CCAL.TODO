from numpy import unique


def guess_type(nu___, n_ca=16):

    if all(float(nu).is_integer() for nu in nu___.ravel()):

        n_un = unique(nu___).size

        if n_un <= 2:

            return "binary"

        elif n_un <= n_ca:

            return "categorical"

    else:

        return "continuous"
