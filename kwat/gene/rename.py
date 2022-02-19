from julia import OnePiece
from numpy import array


def rename(na_):

    na_, ma_ = OnePiece.gene.rename(na_)

    na_ = array(na_)

    na_[ma_ == 2] = None

    return na_
