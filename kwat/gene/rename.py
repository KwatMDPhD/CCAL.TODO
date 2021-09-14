from numpy import full, sort, unique
from pandas import read_csv, read_excel

from ..constant import DATA_DIRECTORY_PATH
from ..dataframe import map_to
from ..dictionary import clean
from ._read import _read


def _map_ens_to_gene():

    return map_to(
        read_csv("{}ens.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t"), "Gene name"
    )


def _map_hgnc_to_gene():
    return map_to(
        _read(None).drop(
            labels=[
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
            axis=1,
        ),
        "symbol",
        fu=_split,
    )


def _map_cg_to_gene():

    cg_ge = {}

    for cg_st in [
        read_excel(
            "{}illumina_humanmethylation27_content.xlsx".format(DATA_DIRECTORY_PATH),
            usecols=[0, 10],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}humanmethylation450_15017482_v1_2.csv.gz".format(DATA_DIRECTORY_PATH),
            skiprows=7,
            usecols=[0, 21],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}infinium_methylationepic_v_1_0_b5_manifest_file.csv.gz".format(
                DATA_DIRECTORY_PATH
            ),
            skiprows=7,
            usecols=[0, 15],
            index_col=0,
            squeeze=True,
        ),
    ]:

        for cg, st in cg_st.dropna().iteritems():

            cg_ge[cg] = st.split(sep=";", maxsplit=1)[0]

    return cg_ge


def rename(na_, ke=True):

    n_na = len(na_)

    ge_ = full(n_na, "", dtype=object)

    an_ge = clean(
        {
            **_map_hgnc_to_gene(),
            **_map_ens_to_gene(),
            **_map_cg_to_gene(),
        }
    )

    n_su = 0

    fa_ = []

    for ie, na in enumerate(na_):

        if na.startswith("ENST") or na.startswith("ENSG"):

            na = na.split(sep=".", maxsplit=1)[0]

        if na in an_ge:

            ge = an_ge[na]

            n_su += 1

        else:

            if ke:

                ge = na

            else:

                ge = None

            fa_.append(na)

        ge_[ie] = ge

    fa_ = sort(unique(fa_))

    n_fa = fa_.size

    print(
        "Renamed {} ({:.2%}) failed {} ({:.2%})".format(
            n_su, n_su / n_na, n_fa, n_fa / n_na
        )
    )

    return ge_, fa_
