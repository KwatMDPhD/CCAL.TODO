from julia import Gene
from numpy import array


def rename(na_):

    na_, ma_ = Gene.rename(na_)

    na_ = array(na_)

    na_[ma_ == 2] = None

    return na_
