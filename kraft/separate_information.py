from numpy import nan
from pandas import DataFrame, concat

from .BAD_STR import BAD_STR
from .binarize import binarize
from .guess_type import guess_type


def separate_information(information_x_sample, bad_values=BAD_STR):

    continuous = []

    binary = []

    for _, values in information_x_sample.iterrows():

        values = values.replace(bad_values, nan)

        if 1 < values.dropna().unique().size:

            try:

                is_continuous = guess_type(values.astype(float)) == "continuous"

            except ValueError:

                is_continuous = False

            if is_continuous:

                continuous.append(values)

            else:

                binary.append(binarize(values))

    if 0 < len(continuous):

        continuous = DataFrame(continuous)

        continuous.index.name = information_x_sample.index.name

    else:

        continuous = None

    if 0 < len(binary):

        for dataframe in binary:

            dataframe.index = (
                "{}.{}".format(dataframe.index.name, str_) for str_ in dataframe.index
            )

        binary = concat(binary)

        binary.index.name = information_x_sample.index.name

    else:

        binary = None

    return continuous, binary
