from numpy import unique


def guess_type(ar, ma=16):

    if all(float(nu).is_integer() for nu in ar.ravel()):

        n_ca = unique(ar).size

        if n_ca <= 2:

            return "binary"

        elif n_ca <= ma:

            return "categorical"

    return "continuous"
