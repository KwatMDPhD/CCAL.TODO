from numpy import nan
from pandas import DataFrame, concat

from .BAD_VALUES import BAD_VALUES
from .get_data_type import get_data_type
from .make_binary_dataframe_from_categorical_series import (
    make_binary_dataframe_from_categorical_series,
)


def isolate_information_x_sample(information_x_sample, bad_values=BAD_VALUES):

    continuouses = []

    binaries = []

    for information, values in information_x_sample.iterrows():

        values = values.replace(bad_values, nan)

        if 1 < values.dropna().unique().size:

            try:

                is_continuous = get_data_type(values.astype(float)) == "continuous"

            except ValueError as exception:

                print(f"{information} is not continuous ({exception}).")

                is_continuous = False

            if is_continuous:

                continuouses.append(values)

            else:

                binaries.append(make_binary_dataframe_from_categorical_series(values))

    if 0 < len(continuouses):

        continuous_information_x_sample = DataFrame(continuouses)

    else:

        continuous_information_x_sample = None

    if 0 < len(binaries):

        binary_information_x_sample = concat(binaries)

    else:

        binary_information_x_sample = None

    return continuous_information_x_sample, binary_information_x_sample
