from numpy import diff


def check_is_sorted(nu_):

    di_ = diff(nu_)

    return (di_ <= 0).all() or (0 <= di_).all()
