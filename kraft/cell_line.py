from pandas import read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


# TODO: use lowercase
def rename(names):

    name_to_aht = read_csv(
        "{}/cell_line_name_aht.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()

    renames = []

    fails = []

    for name in names:

        if name in name_to_aht:

            renames.append(name_to_aht[name])

        else:

            renames.append(name)

            fails.append(name)

    if 0 < len(fails):

        print("Failed {}.".format(sorted(set(fails))))

    return renames
