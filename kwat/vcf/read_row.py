from .ANN import ANN
from .COLUMN import COLUMN


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


def _get_genotype(re, al, gt):

    return [
        [re, *al.split(sep=",")][int(sp)] for sp in gt.replace("/", "|").split(sep="|")
    ]


def _extend(div):

    re = div["REF"]

    al = div["ALT"]

    st, en = _get_variant_start_and_end(int(div["POS"]), re, al)

    div["start_position"] = st

    div["end_position"] = en

    if "CAF" in div:

        div["population_allelic_frequencies"] = [
            float(ca) for ca in div["CAF"].split(sep=",")
        ]

    for dis in div["sample"].values():

        if "GT" in dis:

            dis["genotype"] = _get_genotype(re, al, dis["GT"])

        if "AD" in dis and "DP" in dis:

            dis["allelic_frequency"] = [
                int(ad) / int(dis["DP"]) for ad in dis["AD"].split(sep=",")
            ]


def read_row(st_, n_an=None):

    div = {co: st_[ie] for ie, co in enumerate(COLUMN[: COLUMN.index("FILTER") + 1])}

    no_ = []

    for io in st_[COLUMN.index("INFO")].split(sep=";"):

        if "=" in io:

            kei, vai = io.split(sep="=")

            if kei == "ANN":

                div["ANN"] = []

                for an in vai.split(sep=",")[:n_an]:

                    an_ = an.split(sep="|")

                    div["ANN"].append(
                        {kea: an_[iea + 1] for iea, kea in enumerate(ANN[1:])}
                    )

            else:

                div[kei] = vai

        else:

            no_.append(io)

    if 0 < len(no_):

        div["info_without_field"] = ";".join(no_)

    ief = COLUMN.index("FORMAT")

    fo_ = st_[ief].split(sep=":")

    div["sample"] = []

    for ies, sa in enumerate(st_[ief + 1 :]):

        div["sample"].append({fo: sav for fo, sav in zip(fo_, sa.split(sep=":"))})

    _extend(div)

    return div
