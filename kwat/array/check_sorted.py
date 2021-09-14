from numpy import diff


def check_sorted(nu___):

    di_ = diff(nu___.ravel())

    return (di_ <= 0).all() or (0 <= di_).all()
