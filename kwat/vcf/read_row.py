def _get_variant_start_and_end(po, re, al):

    if len(re) == len(al):

        st = po

        en = po + len(al) - 1

    elif len(re) < len(al):

        st = po

        en = po + 1

    else:

        st = po + 1

        en = po + len(re) - len(al)

    return st, en


def _extend(vd):

    re = vd["REF"]

    al = vd["ALT"]

    st, en = _get_variant_start_and_end(int(vd["POS"]), re, al)

    vd["start_position"] = st

    vd["end_position"] = en

    if "CAF" in vd:

        vd["population_allelic_frequencies"] = [
            float(ca) for ca in vd["CAF"].split(sep=",")
        ]

    for sa in vd["sample"].values():

        if "GT" in sa:

            sa["genotype"] = _get_genotype(re, al, sa["GT"])

        if "AD" in sa and "DP" in sa:

            sa["allelic_frequency"] = [
                int(ad) / int(sa["DP"]) for ad in sa["AD"].split(sep=",")
            ]


from .ANN_KEY import ANN_KEY
from .COLUMN import COLUMN


def read_row(se, n_ioan=None):

    vd = {co: se[ie] for ie, co in enumerate(COLUMN[: COLUMN.index("FILTER") + 1])}

    inno_ = []

    for io in se[COLUMN.index("INFO")].split(sep=";"):

        if "=" in io:

            iofi, iova = io.split(sep="=")

            if iofi == "ANN":

                vd["ANN"] = {}

                for iean, an in enumerate(iova.split(sep=",")[:n_ioan]):

                    anva_ = an.split(sep="|")

                    vd["ANN"][iean] = {
                        anfi: anva_[ieanfi + 1]
                        for ieanfi, anfi in enumerate(ANN_KEY[1:])
                    }

            else:

                vd[iofi] = iova

        else:

            inno_.append(io)

    if 0 < len(inno_):

        vd["info_without_field"] = ";".join(inno_)

    iefo = COLUMN.index("FORMAT")

    fofi_ = se[iefo].split(sep=":")

    vd["sample"] = {}

    for iesa, sa in enumerate(se[iefo + 1 :]):

        vd["sample"][iesa] = {
            fofi: sava for fofi, sava in zip(fofi_, sa.split(sep=":"))
        }

    _extend(vd)

    return vd
