from numpy import nan
from pandas import DataFrame, Index


def read_gmt(gmt_file_path):

    indices = []

    lines = []

    with open(gmt_file_path) as io:

        for line in io:

            split = line.strip().split(sep="\t")

            indices.append(split.pop(0))

            split.pop(0)

            lines.append(sorted(split))

    dataframe = DataFrame(lines, index=Index(indices, name="Gene Set")).fillna(nan)

    return dataframe
