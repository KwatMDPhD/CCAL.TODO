from numpy import asarray, diff


def check_is_sorted(ve):

    di_ = diff(asarray(ve, float))

    return (di_ <= 0).all() or (0 <= di_).all()
