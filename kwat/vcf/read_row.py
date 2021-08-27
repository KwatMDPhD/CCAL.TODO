from ._extend import _extend
from .ANN_KEYS import ANN_KEYS
from .COLUMNS import COLUMNS


def read_row(se, n_ioan=None):

    vd = {co: se[ie] for ie, co in enumerate(COLUMNS[: COLUMNS.index("FILTER") + 1])}

    inno_ = []

    for io in se[COLUMNS.index("INFO")].split(sep=";"):

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

    if 0 < len(inno_):

        vd["info_without_field"] = ";".join(inno_)

    iefo = COLUMNS.index("FORMAT")

    fofi_ = se[iefo].split(sep=":")

    vd["sample"] = {}

    for iesa, sa in enumerate(se[iefo + 1 :]):

        vd["sample"][iesa] = {
            fofi: sava for fofi, sava in zip(fofi_, sa.split(sep=":"))
        }

    _extend(vd)

    return vd
