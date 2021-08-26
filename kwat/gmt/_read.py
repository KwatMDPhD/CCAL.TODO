def _read(pa):

    se_ge_ = {}

    with open(pa) as io:

        for li in io.readlines():

            sp_ = li.strip().split(sep="\t")

            ge_ = [sp for sp in sp_[2:] if sp != ""]

            if 0 < len(ge_):

                se_ge_[sp_[0]] = ge_

    return se_ge_
