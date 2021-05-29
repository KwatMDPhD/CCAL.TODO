from json import (
    dump,
    load,
)


def read(
    file_path,
):

    with open(file_path) as io:

        return load(io)


def write(
    file_path,
    dict,
    indent_n=2,
):

    with open(
        file_path,
        mode="w",
    ) as io:

        dump(
            dict,
            io,
            indent=indent_n,
        )
