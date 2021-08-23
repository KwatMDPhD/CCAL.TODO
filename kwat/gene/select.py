from ..iterable import flatten
from ._map_family import _map_family
from ._read_hgnc import _read_hgnc


def select(
    co_se=None,
    su_=(
        "ribosom",
        "mitochondria",
        "small nucleolar rna",
        "nadh:ubiquinone oxidoreductase",
    ),
):

    if co_se is None:

        co_se = {"locus_group": ["protein-coding gene"]}

    ge_ = _read_hgnc(co_se).loc[:, "symbol"].values

    fa_ge_ = {}

    for ge, fa in _map_family().items():

        if fa is not None and any(su in fa.lower() for su in su_):

            if fa in fa_ge_:

                fa_ge_[fa].append(ge)

            else:

                fa_ge_[fa] = [ge]

    print("Removing:")

    for fa, ba_ in sorted(fa_ge_.items(), key=lambda pa: len(pa[1]), reverse=True):

        print("{}\t{}".format(len(ba_), fa))

    ba_ = flatten(fa_ge_.values())

    return [ge for ge in ge_ if ge not in ba_]
