from ..dictionary import summarize


def _name_feature(fe_, pl, da):

    print(pl)

    pl = int(pl[3:])

    if pl in [96, 97, 570]:

        con = "Gene Symbol"

        def fu(na):

            if na != "":

                return na.split(sep=" /// ", maxsplit=1)[0]

    elif pl in [13534]:

        con = "UCSC_RefGene_Name"

        def fu(na):

            return na.split(sep=";", maxsplit=1)[0]

    elif pl in [5175, 11532]:

        con = "gene_assignment"

        def fu(na):

            if isinstance(na, str) and na not in ["", "---"]:

                return na.split(sep=" // ", maxsplit=2)[1]

    elif pl in [2004, 2005, 3718, 3720]:

        con = "Associated Gene"

        def fu(na):

            return na.split(sep=" // ", maxsplit=1)[0]

    elif pl in [10558]:

        con = "Symbol"

        fu = None

    elif pl in [16686]:

        con = "GB_ACC"

        fu = None

    else:

        con = None

        fu = None

    if con is None:

        return fe_

    for co in da.columns.values:

        if co == con:

            print(">> {} <<".format(co))

        else:

            print(co)

    na_ = da.loc[:, con].values

    if callable(fu):

        na_ = [fu(na) for na in na_]

    fe_na = dict(zip(fe_, na_))

    summarize(fe_na)

    return [fe_na.get(fe) for fe in fe_]
