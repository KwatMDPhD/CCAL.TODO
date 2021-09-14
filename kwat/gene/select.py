from ..iterable import flatten
from ._read import _read
from ._split import _split


def _split_get_first(an):

    sp_ = _split(an)

    if 0 < len(sp_):

        return sp_[0]

    return None


def _map_gene_to_family():

    da = _read(None)

    return dict(
        zip(
            da.loc[:, "symbol"],
            (_split_get_first(fa) for fa in da.loc[:, "gene_family"]),
        )
    )


from ._read import _read


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

    ge_ = _read(co_se).loc[:, "symbol"].values

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
