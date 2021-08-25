from numpy import diff


def check_is_sorted(ve):

    di_ = diff(ve)

    return (di_ <= 0).all() or (0 <= di_).all()
