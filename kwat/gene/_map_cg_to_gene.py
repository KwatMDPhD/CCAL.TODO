from pandas import read_csv, read_excel

from ..constant import data_directory


def _map_cg_to_gene():

    cg_ge = {}

    for cg_st in [
        read_excel(
            "{}illumina_humanmethylation27_content.xlsx".format(data_directory),
            usecols=[0, 10],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}humanmethylation450_15017482_v1_2.csv.gz".format(data_directory),
            skiprows=7,
            usecols=[0, 21],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}infinium_methylationepic_v_1_0_b5_manifest_file.csv.gz".format(
                data_directory
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
