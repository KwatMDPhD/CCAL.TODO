from .ANN_KEYS import ANN_KEYS
from .COLUMNS import COLUMNS


def make_variant_dict_from_vcf_row(ro, n_ioan=None):

    vd = {co: ro[ie] for ie, co in enumerate(COLUMNS[: COLUMNS.index("FILTER") + 1])}

    inno_ = []

    for io in ro[COLUMNS.index("INFO")].split(sep=";"):

        if "=" in io:

            iofi, iova = io.split(sep="=")

            if iofi == "ANN":

                vd["ANN"] = {}

                for iean, an in enumerate(iova.split(sep=",")[:n_ioan]):

                    anva_ = an.split(sep="|")

                    vd["ANN"][iean] = {
                        anfi: anva_[ieanfi + 1]
                        for ieanfi, anfi in enumerate(ANN_KEYS[1:])
                    }

            else:

                vd[iofi] = iova

        else:

            inno_.append(io)

    if len(inno_):

        vd["inno_"] = ";".join(inno_)

    iefo = COLUMNS.index("FORMAT")

    fofi_ = ro[iefo].split(sep=":")

    vd["sample"] = {}

    for iesa, sa in enumerate(ro[iefo + 1 :]):

        vd["sample"][iesa] = {
            fofi: sava for fofi, sava in zip(fofi_, sa.split(sep=":"))
        }

    return vd
