def _read(pa):

    se_ge = {}

    with open(pa) as io:

        for li in io.readlines():

            sp_ = li.strip().split("\t")

            se_ge[sp_[0]] = [sp for sp in sp_[2:] if sp != ""]

    return se_ge


def read(pa_):

    se_ge = {}

    for pa in pa_:

        se_ge.update(_read(pa))

    return se_ge
