from numpy import array


def _is_tolerable(errors, tolerance):

    (e_, e) = array(errors)[-2:]

    return ((e_ - e) / e_ < tolerance).all()
