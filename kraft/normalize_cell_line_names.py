from os.path import join

from pandas import read_csv

from .DATA_DIRECTORY_PATH import DATA_DIRECTORY_PATH


def normalize_cell_line_names(cell_line_names):

    cell_line_name_best_cell_line_name = read_csv(
        join(DATA_DIRECTORY_PATH, "cell_line_name_best_cell_line_name.tsv"),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()

    best_cell_line_names = []

    cell_line_names_failed_to_map = []

    for cell_line_name in cell_line_names:

        if cell_line_name in cell_line_name_best_cell_line_name:

            best_cell_line_names.append(
                cell_line_name_best_cell_line_name[cell_line_name]
            )

        else:

            best_cell_line_names.append(cell_line_name)

            cell_line_names_failed_to_map.append(cell_line_name)

    if len(cell_line_names_failed_to_map):

        print(
            "Cell line names failed to map: {}.".format(
                set(cell_line_names_failed_to_map)
            )
        )

    return best_cell_line_names
