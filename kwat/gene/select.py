from ..iterable import flatten
from ._map_gene_to_family import _map_gene_to_family
from ._read_select_hgnc import _read_select_hgnc


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

        co_se = {
            "locus_group": ["protein-coding gene"],
        }

    ge_ = _read_select_hgnc(co_se).loc[:, "symbol"].values

    fa_ba_ = {}

    for ge, fa in _map_gene_to_family().items():

        if fa is not None and any(ba in fa.lower() for ba in ba_):

            if fa in fa_ba_:

                fa_ba_[fa].append(ge)

            else:

                fa_ba_[fa] = [ge]

    print("Removing:")

    for fa, ba_ in sorted(fa_ba_.items(), key=lambda pa: len(pa[1]), reverse=True):

        print("{}\t{}".format(len(ba_), fa))

    ba_ = flatten(fa_ba_.values())

    return [ge for ge in ge_ if ge not in ba_]
