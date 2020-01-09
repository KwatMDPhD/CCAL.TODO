from re import match, sub

from numpy import fromfile
from pandas import DataFrame

from .cast_builtin import cast_builtin


def read_fcs(fcs_file_path, print_header=False):

    meta_dict = {}

    io = open(fcs_file_path, mode="rb")

    if print_header:

        print(io.read(58))

    io.seek(3)

    meta_dict["FCSVersion"] = io.read(3).decode()

    io.seek(10)

    for key in (
        "$BEGINTEXT",
        "$ENDTEXT",
        "$BEGINDATA",
        "$ENDDATA",
        "$BEGINANALYSIS",
        "$ENDANALYSIS",
    ):

        meta_dict[key] = int(io.read(8))

    separator = io.read(1).decode()

    text = (
        io.read(meta_dict["$ENDTEXT"] - meta_dict["$BEGINTEXT"])
        .decode()
        .split(separator)
    )

    parameter_dict = {}

    for key, value in zip(text[::2], text[1::2]):

        value = cast_builtin(value)

        if key in meta_dict:

            if meta_dict[key] != value:

                print("{}: (header) {} != (text) {}".format(key, meta_dict[key], value))

        if match("\$P[0-9]+[BENRS]", key):

            parameter_index = int(key[2:-1])

            parameter_information = sub("[0-9]+", "n", key)

            if parameter_index not in parameter_dict:

                parameter_dict[parameter_index] = {}

            parameter_dict[parameter_index][parameter_information] = value

        else:

            meta_dict[key] = value

    information_x_parameter = DataFrame(parameter_dict)

    information_x_parameter.columns.name = "ID"

    information_x_parameter = (
        information_x_parameter.T.sort_index().reset_index().set_index("$PnS").T
    )

    information_x_parameter.index.name = "Information"

    io.seek(meta_dict["$BEGINDATA"])

    if meta_dict["$BYTEORD"] == "1,2,3,4":

        byte_order = "<"

    elif meta_dict["$BYTEORD"] == "4,3,2,1":

        byte_order = ">"

    byte_type = meta_dict["$DATATYPE"].lower()

    byte_size = information_x_parameter.loc["$PnB"][0] // 8

    parameter_x_cell = DataFrame(
        fromfile(
            io,
            dtype="{}{}{}".format(byte_order, byte_type, byte_size),
            count=meta_dict["$TOT"] * meta_dict["$PAR"],
        )
        .byteswap()
        .newbyteorder()
        .reshape((meta_dict["$TOT"], meta_dict["$PAR"])),
        columns=information_x_parameter.columns,
    ).T

    parameter_x_cell.columns.name = "Cell"

    io.close()

    return meta_dict, information_x_parameter, parameter_x_cell
