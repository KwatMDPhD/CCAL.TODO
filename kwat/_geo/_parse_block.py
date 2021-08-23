from pandas import DataFrame


def _parse_block(
    block,
):

    dict_table = block.split("_table_begin\n", 1)

    dict = {}

    for line in dict_table[0].splitlines()[:-1]:

        (key, value) = line[1:].split(" = ", 1)

        if key in dict:

            key_original = key

            index = 2

            while key in dict:

                key = "{}_{}".format(key_original, index)

                index += 1

        dict[key] = value

    if len(dict_table) == 2:

        row_ = tuple(line.split("\t") for line in dict_table[1].splitlines()[:-1])

        dict["table"] = DataFrame(
            data=(row[1:] for row in row_[1:]),
            index=(row[0] for row in row_[1:]),
            columns=row_[0][1:],
        )

    return dict
