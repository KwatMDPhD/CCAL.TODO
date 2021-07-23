from numpy import array, full
from pandas import read_csv, read_excel

from .CONSTANT import DATA_DIRECTORY_PATH
from .dataframe import map_to
from .iterable import flatten


def _read_hgnc(co_se):

    da = read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH),
        "\t",
        low_memory=False,
    )

    if co_se is not None:

        ge_ = da.loc[:, "symbol"].values

        bo_ = full(ge_.size, True)

        for co, se in co_se.items():

            print("Selecting by {}: {}...".format(co, ", ".join(se)))

            bo_ &= array(
                [isinstance(an, str) and an in se for an in da.loc[:, co].values]
            )

            print("{}/{}".format(bo_.sum(), bo_.size))

        da = da.loc[bo_, :]

    return da


def _pr(an):

    if isinstance(an, str):

        return an.split("|")

    return []


def _pr1(an):

    pr_ = _pr(an)

    if 0 < len(pr_):

        return pr_[0]

    return None


def _map_hgnc():
    return map_to(
        _read_hgnc(None).drop(
            [
                "locus_group",
                "locus_type",
                "status",
                "location",
                "location_sortable",
                "gene_family",
                "gene_family_id",
                "date_approved_reserved",
                "date_symbol_changed",
                "date_name_changed",
                "date_modified",
                "pubmed_id",
                "lsdb",
            ],
            1,
        ),
        "symbol",
        fu=_pr,
    )


def _map_ens():

    return map_to(
        read_csv("{}ens.tsv.gz".format(DATA_DIRECTORY_PATH), "\t"), "Gene name"
    )


def _map_cg():

    cg_ge = {}

    for cg2_ge in [
        read_excel(
            "{}illumina_humanmethylation27_content.xlsx".format(DATA_DIRECTORY_PATH),
            usecols=[0, 10],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}HumanMethylation450_15017482_v1-2.csv.gz".format(DATA_DIRECTORY_PATH),
            skiprows=7,
            usecols=[0, 21],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}infinium-methylationepic-v-1-0-b5-manifest-file.csv.gz".format(
                DATA_DIRECTORY_PATH
            ),
            skiprows=7,
            usecols=[0, 15],
            index_col=0,
            squeeze=True,
        ),
    ]:

        for cg, ge in cg2_ge.dropna().iteritems():

            cg_ge[cg] = ge.split(";", 1)[0]

    return cg_ge


def rename(na_):

    an_ge = {**_map_hgnc(), **_map_ens(), **_map_cg()}

    ge_ = [an_ge.get(na) for na in na_]

    bo_ = array([ge is not None for ge in ge_])

    n_to = bo_.size

    n_ge = bo_.sum()

    print("Named {}/{} ({:.2%})".format(n_ge, n_to, n_ge / n_to))

    if n_ge == 0:

        return na_

    else:

        return ge_


def _map_family():

    da = _read_hgnc(None)

    return dict(zip(da.loc[:, "symbol"], (_pr1(fa) for fa in da.loc[:, "gene_family"])))


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
