from numpy import floating, integer, ndarray


def read_where_and_map_column_name_on_hdf5_table(hdf5_table, query):

    print(f"Reading {hdf5_table.name} where {query} ...")

    columns = hdf5_table.colnames

    dicts = []

    for row in hdf5_table.read_where(query):

        dict_ = {}

        for column, value in zip(columns, row):

            if isinstance(value, integer):

                value = int(value)

            elif isinstance(value, floating):

                value = float(value)

            elif isinstance(value, ndarray):

                value = value.tolist()

            else:

                value = value.decode()

            dict_[column] = value

        dicts.append(dict_)

    return dicts
