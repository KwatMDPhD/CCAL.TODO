def _read(pa):

    se_ge_ = {}

    with open(pa) as io:

        for li in io.readlines():

            sp_ = li.strip().split(sep="\t")

            se_ge_[sp_[0]] = [sp for sp in sp_[2:] if sp != ""]

    return se_ge_
