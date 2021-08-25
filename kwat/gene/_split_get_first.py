from ._split import _split


def _split_get_first(an):

    sp_ = _split(an)

    if 0 < len(sp_):

        return sp_[0]

    return None
