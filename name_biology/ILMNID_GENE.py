from pandas import read_csv

from .DATA_DIRECTORY_PATH import DATA_DIRECTORY_PATH

ILMNID_GENE = (
    read_csv(
        "{}/HumanMethylation450_15017482_v1-2.csv.gz".format(DATA_DIRECTORY_PATH),
        skiprows=7,
        usecols=(0, 21,),
        index_col=0,
        squeeze=True,
    )
    .dropna()
    .apply(lambda str_: str_.split(";")[0])
    .to_dict()
)
