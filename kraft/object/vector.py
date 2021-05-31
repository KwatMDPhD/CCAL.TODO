from numpy import asarray, diff


def check_is_sorted(nu_):

    di_ = diff(asarray(nu_, float))

    return (di_ <= 0).all() or (0 <= di_).all()
