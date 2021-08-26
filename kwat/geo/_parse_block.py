from pandas import DataFrame


def _parse_block(bl):

    ke_va = {}

    if "_table_begin" in bl:

        ma, ta = bl.split(sep="_table_begin\n")

        ro_ = [li.split(sep="\t") for li in ta.splitlines()[:-1]]

        ke_va["table"] = DataFrame(
            data=[row[1:] for row in ro_[1:]],
            index=[row[0] for row in ro_[1:]],
            columns=ro_[0][1:],
        )

    else:

        ma = bl

    # TODO: check -1
    for li in ma.splitlines()[:-1]:

        ke, va = li[1:].split(sep=" = ", maxsplit=1)

        if ke in ke_va:

            keo = ke

            ie = 2

            while ke in ke_va:

                ke = "{}_{}".format(keo, ie)

                ie += 1

        ke_va[ke] = va

    return ke_va
