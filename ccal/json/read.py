from json import load


def read(pa):
    with open(pa) as io:
        return load(io)
