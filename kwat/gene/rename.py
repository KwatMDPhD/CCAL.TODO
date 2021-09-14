from re import search

from pandas import read_csv, read_excel

from ..constant import DATA_DIRECTORY_PATH
from ..dataframe import map_to
from ..dictionary import clean, rename as dictionary_rename
from ._read import _read
from ._split import _split


def rename(na_, **ke_va):

    na_ = [na.split(sep=".", maxsplit=1)[0] for na in na_ if search(r"^ENS[TG]", na)]

    na_re = {}

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

            na_re[cg] = st.split(sep=";", maxsplit=1)[0]

    na_re.update(
        map_to(
            read_csv("{}ens.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t"), "Gene name"
        )
    )

    na_re.update(
        map_to(
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
    )

    return dictionary_rename(na_, clean(na_re), **ke_va)
