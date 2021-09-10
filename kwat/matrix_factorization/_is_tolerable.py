from numpy import array


def _is_tolerable(er_, to):

    e_, e = array(er_)[-2:]

    return ((e_ - e) / e_ < to).all()
