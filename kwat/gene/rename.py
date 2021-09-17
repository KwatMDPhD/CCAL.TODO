from re import search

from pandas import read_csv, read_excel

from ..constant import DATA_DIRECTORY_PATH
from ..dataframe import map_to
from ..dictionary import clean, rename as dictionary_rename
from ..string import split_and_get
from ._read import _read


def _split(an):

    if isinstance(an, str):

        return an.split(sep="|")

    else:

        return []


def rename(na_, **di):

    na_re = {}

    for cg_re in [
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

        for cg, re in cg_re.dropna().iteritems():

            na_re[cg] = split_and_get(re, ";", 0)

    na_re.update(
        map_to(
            read_csv("{}ens.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t"), "Gene name"
        )
    )

    na_re.update(
        map_to(
            _read({}).drop(
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

    return dictionary_rename(
        [split_and_get(na, ".", 0) for na in na_ if search(r"^ENS[TG]", na)],
        clean(na_re),
        **di
    )
