from pandas import read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def rename(names):

    name_to_aht = read_csv(
        "{}/cell_line_name_aht.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()

    ahts = []

    names_failed = []

    for name in names:

        if name in name_to_aht:

            ahts.append(name_to_aht[name])

        else:

            ahts.append(name)

            names_failed.append(name)

    if 0 < len(names_failed):

        print("Failed to map {}.".format(sorted(set(names_failed))))

    return ahts
