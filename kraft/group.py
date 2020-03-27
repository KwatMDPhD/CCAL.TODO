def group(dataframe):

    print("Grouping index with median...")

    print(dataframe.shape)

    dataframe = dataframe.groupby(level=0).median()

    print(dataframe.shape)

    return dataframe
