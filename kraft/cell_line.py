from pandas import read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def rename(names):

    name_to_rename = read_csv(
        "{}/cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()

    renames = []

    fails = []

    for name in names:

        name_lower = name.lower()

        if name_lower in name_to_rename:

            renames.append(name_to_rename[name_lower])

        else:

            renames.append(name)

            fails.append(name)

    if 0 < len(fails):

        print("Failed {}.".format(sorted(set(fails))))

    return renames
