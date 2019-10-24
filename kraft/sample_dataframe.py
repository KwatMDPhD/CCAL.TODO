from math import floor

from numpy.random import choice


def sample_dataframe(dataframe, axis0_size, axis1_size):

    assert axis0_size is not None or axis1_size is not None

    if axis0_size is not None and axis1_size is not None:

        return dataframe.loc[
            choice(
                dataframe.index,
                size=int(floor(dataframe.shape[0] * axis0_size)),
                replace=False,
            ),
            choice(
                dataframe.columns,
                size=int(floor(dataframe.shape[1] * axis1_size)),
                replace=False,
            ),
        ]

    elif axis0_size is not None:

        return dataframe.loc[
            choice(
                dataframe.index,
                size=int(floor(dataframe.shape[0] * axis0_size)),
                replace=False,
            ),
        ]

    else:

        return dataframe[
            choice(
                dataframe.columns,
                size=int(floor(dataframe.shape[1] * axis1_size)),
                replace=False,
            )
        ]
