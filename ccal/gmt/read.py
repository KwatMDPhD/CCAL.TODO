def _read(pa):
    se_ge_ = {}

    with open(pa) as io:
        for li in io.readlines():
            sp_ = li.strip().split(sep="\t")

            ge_ = [sp for sp in sp_[2:] if sp != ""]

            se = sp_[0]

            if len(ge_) == 0:
                print("{} has 0 gene.".format(se))

            else:
                se_ge_[se] = ge_

    return se_ge_


def read(pa_):
    se_ge_ = {}

    for pa in pa_:
        se_ge_.update(_read(pa))

    return se_ge_
