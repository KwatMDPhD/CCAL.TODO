from ..constant import NUMBER_OF_CATEGORY
from .get_not_nan_unique import get_not_nan_unique


def guess_type(nu___, n_ca=NUMBER_OF_CATEGORY):
    un_ = get_not_nan_unique(nu___)

    if all(float(nu).is_integer() for nu in un_):
        n_un = un_.size

        if n_un <= 2:
            return "binary"

        elif n_un <= n_ca:
            return "categorical"

    else:
        return "continuous"
