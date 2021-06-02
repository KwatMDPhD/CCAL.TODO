from numpy import array, full
from pandas import read_csv, read_excel

from .CONSTANT import DATA_DIRECTORY_PATH
from .object.dataframe import map_to


def _map_hgnc():
    def pr(an):

        if isinstance(an, str):

            return an.split("|")

        return []

    return map_to(
        read_csv(
            "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH),
            "\t",
            low_memory=False,
        ).drop(
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


def select(co_se=None):

    if co_se is None:

        co_se = {"locus_group": ["protein-coding gene"]}

    da = read_csv("{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH), "\t")

    ge_ = da.pop("symbol")

    bo_ = full(ge_.size, True)

    for co, se in co_se.items():

        print("Selecting by {}...".format(co))

        bo_ &= array([isinstance(an, str) and an in se for an in da[co].values])

        print("{}/{}".format(bo_.sum(), bo_.size))

    return ge_[bo_]
