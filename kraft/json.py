from json import dump, load


def read(file_path):

    with open(file_path) as io:

        return load(io)


def write(file_path, dictionary, indent=2):

    with open(file_path, mode="w") as io:

        dump(dictionary, io, indent=indent)
