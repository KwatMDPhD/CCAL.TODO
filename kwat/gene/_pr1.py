from ._pr import _pr


def _pr1(an):

    pr_ = _pr(an)

    if 0 < len(pr_):

        return pr_[0]

    return None
