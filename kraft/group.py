def group(dataframe):

    print("Grouping index with median...")

    print(dataframe.shape)

    if dataframe.shape[0] == 0:

        return dataframe

    dataframe = dataframe.groupby(level=0).median()

    print(dataframe.shape)

    return dataframe
