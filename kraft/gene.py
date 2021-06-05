from numpy import array, full
from pandas import isna, read_csv, read_excel

from .CONSTANT import DATA_DIRECTORY_PATH
from .object.dataframe import map_to


def _read_hgnc():

    return read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH),
        "\t",
        low_memory=False,
    )


def _map_hgnc():
    def pr(an):

        if isinstance(an, str):

            return an.split("|")

        return []

    return map_to(
        _read_hgnc().drop(
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
        fu=pr,
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

    da = _read_hgnc()

    def simplify_family(fa):

        if isna(fa):

            return None

        else:

            return fa.split("|")[0]

    return dict(zip(da["symbol"], (simplify_family(fa) for fa in da["gene_family"])))


def _list_bad(
    fa_=("ribosom", "mitochondria", "small nuclear rna", "small nucleolar rna")
):

    ge_ = []

    for ge, fa in _map_family().items():

        if fa is not None and any(su in fa.lower() for su in fa_):

            ge_.append(ge)

    ge_ = sorted(set(ge_))

    print("{} bad genes.".format(len(ge_)))

    return ge_


def select(co_se=None, re=True):

    if co_se is None:

        co_se = {"locus_group": ["protein-coding gene"]}

    da = _read_hgnc()

    ge_ = da.pop("symbol").values

    bo_ = full(ge_.size, True)

    for co, se in co_se.items():

        print("Selecting by {}: {}...".format(co, ", ".join(se)))

        bo_ &= array([isinstance(an, str) and an in se for an in da[co].values])

        print("{}/{}".format(bo_.sum(), bo_.size))

    if re:

        print("Removing bad...")

        ba_ = _list_bad()

        bo_ &= array([ge not in ba_ for ge in ge_])

        print("{}/{}".format(bo_.sum(), bo_.size))

    return ge_[bo_]
