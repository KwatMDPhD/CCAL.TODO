from numpy import diff


def check_sorted(ar):

    di_ = diff(ar.ravel())

    return (di_ <= 0).all() or (0 <= di_).all()
