from numpy import array


def _is_tolerable(er_it_ma, to):

    er2_, er1_ = array(er_it_ma)[-2:]

    return ((er2_ - er1_) / er2_ <= to).all()
