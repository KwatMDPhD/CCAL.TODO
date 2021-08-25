from pandas import read_csv, read_excel

from ..constant import DATA_DIRECTORY


def _map_cg_to_gene():

    cg_ge = {}

    for cg_st in [
        read_excel(
            "{}illumina_humanmethylation27_content.xlsx".format(DATA_DIRECTORY),
            usecols=[0, 10],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}HumanMethylation450_15017482_v1-2.csv.gz".format(DATA_DIRECTORY),
            skiprows=7,
            usecols=[0, 21],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}infinium-methylationepic-v-1-0-b5-manifest-file.csv.gz".format(
                DATA_DIRECTORY
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
