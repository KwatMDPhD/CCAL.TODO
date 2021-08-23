from numpy import array, diff


def check_is_sorted(ve):

    di_ = diff(array(ve, float))

    return (di_ <= 0).all() or (0 <= di_).all()
