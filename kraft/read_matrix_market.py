from pandas import DataFrame, read_csv
from scipy.io import mmread


def read_matrix_market(
    matrix_mtx_file_path,
    index_file_path,
    column_file_path,
    index_name=None,
    column_name=None,
):

    dataframe = DataFrame(
        mmread(matrix_mtx_file_path).toarray(),
        index=read_csv(index_file_path, sep="\t", header=None).iloc[:, -1],
        columns=read_csv(column_file_path, sep="\t", header=None, squeeze=True),
    )

    if dataframe.index.has_duplicates:

        print("Index duplicated. Merging duplicates with median ...")

        dataframe = dataframe.groupby(level=0).median()

    dataframe.sort_index(inplace=True)

    dataframe.index.name = index_name

    if dataframe.columns.has_duplicates:

        print("Column duplicated. Merging duplicates with median ...")

        dataframe = dataframe.T.groupby(level=0).median().T

    dataframe.sort_index(axis=1, inplace=True)

    dataframe.columns.name = column_name

    return dataframe
