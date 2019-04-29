def split_dataframe(dataframe, axis, n_split):

    n = dataframe.shape[axis] // n_split

    dataframes = []

    for i in range(n_split):

        start_i = i * n

        end_i = (i + 1) * n

        if axis == 0:

            dataframes.append(dataframe.iloc[start_i:end_i])

        elif axis == 1:

            dataframes.append(dataframe.iloc[:, start_i:end_i])

    i = n * n_split

    if i < dataframe.shape[axis]:

        if axis == 0:

            dataframes.append(dataframe.iloc[i:])

        elif axis == 1:

            dataframes.append(dataframe.iloc[:, i:])

    return dataframes
