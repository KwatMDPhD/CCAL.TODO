from numpy import unique


def make_axis_different(dataframes, axis):

    elements = []

    for dataframe in dataframes:

        if axis == 0:

            elements += dataframe.index.tolist()

        else:

            elements += dataframe.columns.tolist()

    elements, counts = unique(elements, return_counts=True)

    elements_to_drop = elements[1 < counts]

    print("Dropping {} axis-{} elements...".format(elements_to_drop.size, axis))

    return tuple(
        dataframe.drop(elements_to_drop, axis=axis, errors="ignore")
        for dataframe in dataframes
    )
