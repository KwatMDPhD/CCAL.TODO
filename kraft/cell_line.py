from pandas import read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def normalize_cell_line_names(names):

    name_aht = read_csv(
        "{}/cell_line_name_aht.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()

    ahts = []

    names_failed = []

    for name in names:

        if name in name_aht:

            ahts.append(name_aht[name])

        else:

            ahts.append(name)

            names_failed.append(name)

    if 0 < len(names_failed):

        print("Failed to map {}.".format(set(names_failed)))

    assert len(ahts) == len(set(ahts))

    return ahts
