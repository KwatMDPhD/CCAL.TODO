from numpy import array

from ..iterable import flatten
from ._read import _read
from .map_gene_to_family import map_gene_to_family


def _get_value_size(pa):

    return len(pa[1])


def select(
    co_se=None,
    ba_=(
        "ribosom",
        "mitochondria",
        "small nucleolar rna",
        "nadh:ubiquinone oxidoreductase",
    ),
):

    if co_se is None:

        co_se = {"locus_group": ["protein-coding gene"]}

    ge_ = _read(co_se).loc[:, "symbol"].values

    if 0 < len(ba_):

        fa_ba_ = {}

        for ge, fa in map_gene_to_family().items():

            if fa is not None and any(ba in fa.lower() for ba in ba_):

                if fa in fa_ba_:

                    fa_ba_[fa].append(ge)

                else:

                    fa_ba_[fa] = [ge]

        print("Removing:")

        for fa, ba_ in sorted(fa_ba_.items(), key=_get_value_size, reverse=True):

            print("{}\t{}".format(len(ba_), fa))

        ba_ = flatten(fa_ba_.values())

        return array([ge for ge in ge_ if ge not in ba_])

    else:

        return ge_
